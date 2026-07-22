from __future__ import annotations

import contextlib
import math
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps


class VideoBackendError(RuntimeError):
    """Raised when no usable decoder/encoder is available."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    backend: str


def _import_av():
    try:
        import av  # type: ignore
    except ImportError:
        return None
    return av


def find_ffmpeg(explicit: str | Path | None = None) -> str | None:
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        resolved_explicit = shutil.which(str(explicit))
        if resolved_explicit:
            return resolved_explicit
        raise FileNotFoundError(f"Explicit ffmpeg binary does not exist: {candidate}")
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    try:
        import imageio_ffmpeg  # type: ignore

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    except (ImportError, RuntimeError, OSError):
        pass
    return None


def available_backend(requested: str = "auto", *, ffmpeg: str | Path | None = None) -> str:
    if requested not in {"auto", "pyav", "ffmpeg"}:
        raise VideoBackendError(f"Unsupported video backend: {requested!r}")
    if requested in {"auto", "pyav"} and _import_av() is not None:
        return "pyav"
    if requested == "pyav":
        raise VideoBackendError(
            "The PyAV backend was requested but package 'av' is unavailable. "
            "Install PyAV or use --backend ffmpeg."
        )
    if find_ffmpeg(ffmpeg) is not None:
        return "ffmpeg"
    raise VideoBackendError(
        "No video backend is available. Install PyAV (`pip install av`) or make the "
        "ffmpeg executable available on PATH (or pass --ffmpeg /path/to/ffmpeg)."
    )


def _fraction_fps(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _probe_pyav(path: Path) -> VideoProbe:
    av = _import_av()
    if av is None:
        raise VideoBackendError("PyAV is unavailable.")
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise VideoBackendError(f"Video has no visual stream: {path}")
            stream = container.streams.video[0]
            fps = _fraction_fps(stream.average_rate) or _fraction_fps(stream.base_rate)
            count = int(stream.frames) if int(stream.frames or 0) > 0 else None
            return VideoProbe(
                width=int(stream.codec_context.width),
                height=int(stream.codec_context.height),
                fps=fps,
                frame_count=count,
                backend="pyav",
            )
    except VideoBackendError:
        raise
    except Exception as exc:
        raise VideoBackendError(f"PyAV could not inspect {path}: {exc}") from exc


_SIZE_RE = re.compile(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)")
_FPS_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*fps\b")
_TBR_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*tbr\b")


def _probe_ffmpeg(path: Path, ffmpeg: str) -> VideoProbe:
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = "\n".join([completed.stdout, completed.stderr])
    video_lines = [line for line in output.splitlines() if "Video:" in line]
    if not video_lines:
        raise VideoBackendError(f"ffmpeg could not find a video stream in {path}.\n{output[-1200:]}")
    line = video_lines[0]
    size = _SIZE_RE.search(line)
    if size is None:
        raise VideoBackendError(f"ffmpeg did not report video dimensions for {path}: {line}")
    fps_match = _FPS_RE.search(line) or _TBR_RE.search(line)
    fps = float(fps_match.group(1)) if fps_match else None
    return VideoProbe(
        width=int(size.group(1)),
        height=int(size.group(2)),
        fps=fps,
        frame_count=None,
        backend="ffmpeg",
    )


def probe_video(
    path: str | Path,
    *,
    backend: str = "auto",
    ffmpeg: str | Path | None = None,
) -> VideoProbe:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    selected = available_backend(backend, ffmpeg=ffmpeg)
    if selected == "pyav":
        try:
            return _probe_pyav(source)
        except VideoBackendError:
            if backend != "auto":
                raise
            binary = find_ffmpeg(ffmpeg)
            if binary is None:
                raise
            return _probe_ffmpeg(source, binary)
    binary = find_ffmpeg(ffmpeg)
    assert binary is not None
    return _probe_ffmpeg(source, binary)


def _iter_pyav_frames(
    path: Path,
    *,
    start_seconds: float,
    max_frames: int | None,
) -> Iterator[Image.Image]:
    av = _import_av()
    if av is None:
        raise VideoBackendError("PyAV is unavailable.")
    container = av.open(str(path), mode="r")
    try:
        if not container.streams.video:
            raise VideoBackendError(f"Video has no visual stream: {path}")
        stream = container.streams.video[0]
        if start_seconds > 0 and stream.time_base is not None:
            timestamp = int(start_seconds / float(stream.time_base))
            container.seek(timestamp, stream=stream, any_frame=False, backward=True)
        yielded = 0
        tolerance = 1e-7
        for frame in container.decode(stream):
            if frame.pts is not None and frame.time_base is not None:
                frame_seconds = float(frame.pts * frame.time_base)
                if frame_seconds + tolerance < start_seconds:
                    continue
            yield Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        container.close()


def _read_exact(handle, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = handle.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _iter_ffmpeg_frames(
    path: Path,
    *,
    probe: VideoProbe,
    ffmpeg: str,
    start_seconds: float,
    max_frames: int | None,
) -> Iterator[Image.Image]:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path)]
    if start_seconds > 0:
        # Accurate output-side seek: slower than input-side seeking, but avoids
        # keyframe-dependent off-by-one alignment for episode video shards.
        command.extend(["-ss", f"{start_seconds:.9f}"])
    command.extend(["-map", "0:v:0", "-an", "-sn", "-dn", "-vsync", "0"])
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.extend(["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])

    frame_bytes = probe.width * probe.height * 3
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=error_log,
        )
        assert process.stdout is not None
        completed_normally = False
        try:
            while True:
                payload = _read_exact(process.stdout, frame_bytes)
                if not payload:
                    break
                if len(payload) != frame_bytes:
                    raise VideoBackendError(
                        f"ffmpeg returned a truncated raw frame from {path}: "
                        f"{len(payload)} of {frame_bytes} bytes."
                    )
                yield Image.frombytes("RGB", (probe.width, probe.height), payload)
            completed_normally = True
        finally:
            process.stdout.close()
            if not completed_normally and process.poll() is None:
                process.terminate()
            return_code = process.wait()
            if completed_normally and return_code != 0:
                error_log.seek(0)
                details = error_log.read().decode("utf-8", errors="replace")[-2000:]
                raise VideoBackendError(f"ffmpeg failed while decoding {path}:\n{details}")


def iter_video_frames(
    path: str | Path,
    *,
    backend: str = "auto",
    ffmpeg: str | Path | None = None,
    start_seconds: float = 0.0,
    max_frames: int | None = None,
) -> tuple[VideoProbe, Iterator[Image.Image]]:
    source = Path(path).expanduser().resolve()
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("start_seconds must be finite and non-negative.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided.")
    probe = probe_video(source, backend=backend, ffmpeg=ffmpeg)
    selected = probe.backend
    if selected == "pyav":
        iterator = _iter_pyav_frames(
            source,
            start_seconds=start_seconds,
            max_frames=max_frames,
        )
    else:
        binary = find_ffmpeg(ffmpeg)
        assert binary is not None
        iterator = _iter_ffmpeg_frames(
            source,
            probe=probe,
            ffmpeg=binary,
            start_seconds=start_seconds,
            max_frames=max_frames,
        )
    return probe, iterator


def even_size(width: int, height: int) -> tuple[int, int]:
    return width + width % 2, height + height % 2


class VideoWriter:
    """Transactional MP4 writer with PyAV and ffmpeg implementations."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        backend: str = "auto",
        ffmpeg: str | Path | None = None,
        codec: str = "h264",
        crf: int = 18,
        preset: str = "medium",
        overwrite: bool = False,
    ) -> None:
        if width < 2 or height < 2:
            raise ValueError(f"Video dimensions are too small: {width}x{height}")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be finite and positive.")
        if not 0 <= crf <= 51:
            raise ValueError("CRF must be between 0 and 51.")
        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {self.output_path}. Pass --overwrite to replace it."
            )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex[:10]
        self.temporary_path = self.output_path.with_name(
            f".{self.output_path.stem}.{token}.tmp{self.output_path.suffix}"
        )
        self.source_width = width
        self.source_height = height
        self.width, self.height = even_size(width, height)
        self.fps = float(fps)
        self.requested_backend = backend
        self._ffmpeg_request = ffmpeg
        self.backend = available_backend(backend, ffmpeg=ffmpeg)
        self.ffmpeg = find_ffmpeg(ffmpeg) if self.backend == "ffmpeg" else None
        self.codec = codec
        self.crf = crf
        self.preset = preset
        self.frame_count = 0
        self._closed = False
        self._committed = False
        self._container = None
        self._stream = None
        self._time_base = None
        self._process = None
        self._error_log = None
        self._open()

    def _open(self) -> None:
        if self.backend == "pyav":
            av = _import_av()
            assert av is not None
            container = None
            try:
                container = av.open(
                    str(self.temporary_path),
                    mode="w",
                    options={"movflags": "faststart"},
                )
                rate = Fraction(str(self.fps)).limit_denominator(100_000)
                stream = container.add_stream(
                    self.codec,
                    rate=rate,
                    options={"crf": str(self.crf), "preset": self.preset},
                )
                stream.width = self.width
                stream.height = self.height
                stream.pix_fmt = "yuv420p"
                stream.time_base = Fraction(rate.denominator, rate.numerator)
                self._container = container
                self._stream = stream
                self._time_base = Fraction(rate.denominator, rate.numerator)
                return
            except Exception as exc:
                if container is not None:
                    with contextlib.suppress(Exception):
                        container.close()
                with contextlib.suppress(FileNotFoundError):
                    self.temporary_path.unlink()
                if self.requested_backend == "auto":
                    try:
                        binary = find_ffmpeg(self._ffmpeg_request)
                    except FileNotFoundError:
                        binary = None
                    if binary is not None:
                        self.backend = "ffmpeg"
                        self.ffmpeg = binary
                        self._open()
                        return
                raise VideoBackendError(
                    f"PyAV could not start encoder {self.codec!r}: {exc}"
                ) from exc

        assert self.ffmpeg is not None
        self._error_log = tempfile.TemporaryFile()
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.9g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            self.codec,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.temporary_path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._error_log,
            )
        except OSError as exc:
            self._error_log.close()
            self._error_log = None
            with contextlib.suppress(FileNotFoundError):
                self.temporary_path.unlink()
            raise VideoBackendError(f"Could not start ffmpeg encoder: {exc}") from exc

    def _prepare_frame(self, frame: Image.Image) -> Image.Image:
        if frame.size != (self.source_width, self.source_height):
            raise ValueError(
                f"Frame size changed within an episode: expected "
                f"{self.source_width}x{self.source_height}, got {frame.width}x{frame.height}."
            )
        rgb = frame.convert("RGB")
        if rgb.size != (self.width, self.height):
            rgb = ImageOps.expand(
                rgb,
                border=(0, 0, self.width - rgb.width, self.height - rgb.height),
                fill=(0, 0, 0),
            )
        return rgb

    def write(self, frame: Image.Image) -> None:
        if self._closed:
            raise RuntimeError("Cannot write to a closed video writer.")
        rgb = self._prepare_frame(frame)
        if self.backend == "pyav":
            av = _import_av()
            assert av is not None and self._container is not None and self._stream is not None
            import numpy as np

            video_frame = av.VideoFrame.from_ndarray(np.asarray(rgb), format="rgb24")
            video_frame.pts = self.frame_count
            video_frame.time_base = self._time_base
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)
        else:
            assert self._process is not None and self._process.stdin is not None
            try:
                self._process.stdin.write(rgb.tobytes())
            except BrokenPipeError as exc:
                raise VideoBackendError("ffmpeg encoder closed its input unexpectedly.") from exc
        self.frame_count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.backend == "pyav":
                assert self._container is not None and self._stream is not None
                for packet in self._stream.encode():
                    self._container.mux(packet)
                self._container.close()
            else:
                assert self._process is not None and self._process.stdin is not None
                self._process.stdin.close()
                return_code = self._process.wait()
                if return_code != 0:
                    details = ""
                    if self._error_log is not None:
                        self._error_log.seek(0)
                        details = self._error_log.read().decode("utf-8", errors="replace")[-2000:]
                    raise VideoBackendError(f"ffmpeg encoding failed:\n{details}")
        finally:
            if self._error_log is not None:
                self._error_log.close()
                self._error_log = None

    def commit(self) -> Path:
        self.close()
        if self.frame_count < 1:
            self.abort()
            raise VideoBackendError("Refusing to commit an empty video.")
        if self.output_path.exists():
            self.output_path.unlink()
        self.temporary_path.replace(self.output_path)
        self._committed = True
        return self.output_path

    def abort(self) -> None:
        if not self._closed:
            with contextlib.suppress(Exception):
                if self.backend == "ffmpeg" and self._process is not None:
                    if self._process.stdin is not None:
                        self._process.stdin.close()
                    if self._process.poll() is None:
                        self._process.terminate()
                    self._process.wait(timeout=5)
                elif self.backend == "pyav" and self._container is not None:
                    self._container.close()
            self._closed = True
        with contextlib.suppress(FileNotFoundError):
            self.temporary_path.unlink()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def __del__(self) -> None:
        if not self._committed:
            with contextlib.suppress(Exception):
                self.abort()
