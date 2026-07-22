from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .curves import EpisodeCurve, value_domain
from .overlay import EpisodeOverlayRenderer


def _background(frame_index: int, total_frames: int, size: tuple[int, int]) -> Image.Image:
    width, height = size
    frame = Image.new("RGB", size, (22, 35, 44))
    draw = ImageDraw.Draw(frame, "RGBA")
    for y in range(height):
        amount = y / max(height - 1, 1)
        draw.line(
            (0, y, width, y),
            fill=(
                round(25 + 15 * amount),
                round(43 + 25 * amount),
                round(54 + 22 * amount),
                255,
            ),
        )
    # Soft industrial lights and depth-of-field shapes.
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    glow_draw.ellipse((width * 0.08, -height * 0.25, width * 0.58, height * 0.5), fill=(72, 161, 190, 100))
    glow_draw.ellipse((width * 0.62, height * 0.02, width * 1.02, height * 0.58), fill=(255, 180, 96, 55))
    frame = Image.alpha_composite(frame.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(70)))
    draw = ImageDraw.Draw(frame, "RGBA")

    table_y = round(height * 0.57)
    draw.polygon(
        [(0, table_y), (width, table_y - 25), (width, height), (0, height)],
        fill=(47, 55, 59, 255),
    )
    draw.line((0, table_y, width, table_y - 25), fill=(188, 205, 210, 90), width=3)

    progress = frame_index / max(total_frames - 1, 1)
    object_x = round(width * (0.68 - 0.18 * progress))
    object_y = round(height * 0.52)
    draw.rounded_rectangle(
        (object_x - 42, object_y - 35, object_x + 42, object_y + 35),
        radius=10,
        fill=(216, 130, 63, 255),
        outline=(255, 211, 149, 210),
        width=3,
    )

    # Simple moving robot arm silhouette.
    base = (round(width * 0.23), round(height * 0.61))
    joint = (round(width * (0.36 + 0.07 * math.sin(progress * math.pi))), round(height * 0.36))
    tip = (round(width * (0.55 + 0.09 * progress)), round(height * (0.45 + 0.03 * math.sin(progress * 2 * math.pi))))
    draw.line((base, joint), fill=(206, 221, 226, 255), width=34)
    draw.line((joint, tip), fill=(170, 190, 197, 255), width=28)
    for x, y, radius in [(base[0], base[1], 27), (joint[0], joint[1], 23), (tip[0], tip[1], 16)]:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(84, 105, 113, 255), outline=(224, 239, 243, 220), width=3)
    draw.line((tip[0] - 8, tip[1] + 10, tip[0] - 18, tip[1] + 30), fill=(38, 48, 52, 255), width=7)
    draw.line((tip[0] + 8, tip[1] + 10, tip[0] + 18, tip[1] + 30), fill=(38, 48, 52, 255), width=7)

    draw.rounded_rectangle((20, 20, 235, 55), radius=12, fill=(3, 10, 15, 115))
    draw.text((36, 37), "SYNTHETIC TASK CAMERA", fill=(222, 238, 242, 220), anchor="lm")
    return frame.convert("RGB")


def generate_demo(
    output: str | Path,
    *,
    frames: int = 96,
    gif: bool = False,
    size: tuple[int, int] = (960, 540),
) -> Path:
    if frames < 16:
        raise ValueError("Demo requires at least 16 frames.")
    frame_indices = tuple(range(8, frames - 1))
    values = tuple(
        -0.82
        + 0.62 * ((index - frame_indices[0]) / max(len(frame_indices) - 1, 1))
        + 0.055 * math.sin(index * 0.24)
        for index in frame_indices
    )
    curve = EpisodeCurve(episode_id=7, frame_indices=frame_indices, values=values)
    renderer = EpisodeOverlayRenderer(
        size=size,
        curve=curve,
        timeline_first_frame=0,
        timeline_last_frame=frames - 1,
        y_domain=value_domain([curve]),
        fps=24,
        camera_key="observation.images.front",
        debug_alignment=False,
    )
    rendered: list[Image.Image] = []
    preview_index = round(frames * 0.64)
    preview: Image.Image | None = None
    for ordinal in range(frames):
        image = renderer.render(
            _background(ordinal, frames, size),
            frame_index=ordinal,
            ordinal=ordinal,
            total_frames=frames,
        )
        if ordinal == preview_index:
            preview = image.copy()
        if gif:
            rendered.append(image.resize((640, 360), Image.Resampling.LANCZOS))
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert preview is not None
    if gif:
        rendered[0].save(
            destination,
            save_all=True,
            append_images=rendered[1:],
            duration=42,
            loop=0,
            optimize=False,
        )
    else:
        preview.save(destination, quality=94)
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a dependency-light visual style preview.")
    parser.add_argument("--output", default="episode_value_video_demo.jpg")
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args(argv)
    print(
        generate_demo(
            args.output,
            frames=args.frames,
            gif=args.gif,
            size=(args.width, args.height),
        )
    )


if __name__ == "__main__":
    main()
