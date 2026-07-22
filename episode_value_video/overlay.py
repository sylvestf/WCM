from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .curves import CurveFrameState, EpisodeCurve


def parse_hex_color(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        raise ValueError(f"Color must be #RRGGBB, got {value!r}.")
    try:
        return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
    except ValueError as exc:
        raise ValueError(f"Color must be hexadecimal, got {value!r}.") from exc


def _lerp(left: float, right: float, amount: float) -> float:
    return left + (right - left) * amount


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _truncate_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    ellipsis = "…"
    while text and _text_width(draw, text + ellipsis, font) > max_width:
        text = text[:-1]
    return text + ellipsis if text else ellipsis


def _font_candidates(explicit: str | Path | None, *, bold: bool) -> list[str]:
    candidates: list[str] = []
    if explicit is not None:
        candidates.append(str(Path(explicit).expanduser()))
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/seguisb.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "DejaVuSans.ttf",
            ]
        )
    return candidates


def _load_font(
    size: int,
    *,
    explicit: str | Path | None = None,
    bold: bool = False,
) -> ImageFont.ImageFont:
    for candidate in _font_candidates(explicit, bold=bold):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass(frozen=True, slots=True)
class OverlayTheme:
    accent: tuple[int, int, int] = (97, 228, 255)
    foreground: tuple[int, int, int] = (243, 249, 252)
    muted: tuple[int, int, int] = (166, 184, 194)
    panel: tuple[int, int, int, int] = (4, 11, 17, 186)
    grid: tuple[int, int, int, int] = (196, 218, 228, 35)


@dataclass(frozen=True, slots=True)
class OverlayLayout:
    panel: tuple[int, int, int, int]
    chart: tuple[int, int, int, int]
    progress: tuple[int, int, int, int]
    margin: int
    radius: int


class EpisodeOverlayRenderer:
    """Render a cinematic, fixed-scale value HUD over original episode frames."""

    def __init__(
        self,
        *,
        size: tuple[int, int],
        curve: EpisodeCurve,
        timeline_first_frame: int,
        timeline_last_frame: int,
        y_domain: tuple[float, float],
        fps: float,
        camera_key: str | None = None,
        title: str = "WORLD CRITIC",
        accent: str = "#61E4FF",
        font_path: str | Path | None = None,
        debug_alignment: bool = False,
    ) -> None:
        self.width, self.height = size
        if self.width < 320 or self.height < 180:
            raise ValueError(
                f"Video is too small for the overlay: {self.width}x{self.height}; "
                "minimum is 320x180."
            )
        if timeline_last_frame <= timeline_first_frame:
            raise ValueError("Episode timeline must contain at least two frames.")
        if not y_domain[0] < y_domain[1]:
            raise ValueError(f"Invalid y-domain: {y_domain}")
        self.curve = curve
        self.timeline_first = int(timeline_first_frame)
        self.timeline_last = int(timeline_last_frame)
        self.y_min, self.y_max = map(float, y_domain)
        self.fps = float(fps)
        self.camera_key = camera_key
        self.title = title.strip() or "WORLD CRITIC"
        self.debug_alignment = debug_alignment
        self.compact = self.width < 720 or self.height < 420
        self.ultra_compact = self.width < 480 or self.height < 260
        self.theme = OverlayTheme(accent=parse_hex_color(accent))
        self.scale = max(0.58, min(1.45, min(self.width / 1280.0, self.height / 720.0)))
        self.font_path = font_path
        self.font_tiny = _load_font(max(11, round(13 * self.scale)), explicit=font_path)
        self.font_small = _load_font(max(12, round(15 * self.scale)), explicit=font_path)
        self.font_small_bold = _load_font(
            max(12, round(15 * self.scale)), explicit=font_path, bold=True
        )
        self.font_medium = _load_font(
            max(14, round(20 * self.scale)), explicit=font_path, bold=True
        )
        self.font_value = _load_font(
            max(20, round(32 * self.scale)), explicit=font_path, bold=True
        )
        self.layout = self._build_layout()
        self._static_overlay = self._build_static_overlay()

    def _build_layout(self) -> OverlayLayout:
        margin = max(12, round(min(self.width, self.height) * 0.026))
        panel_ratio = 0.42 if self.ultra_compact else (0.38 if self.compact else 0.315)
        panel_height = max(round(170 * self.scale), round(self.height * panel_ratio))
        panel_height = min(panel_height, self.height - 2 * margin)
        panel = (margin, self.height - panel_height - margin, self.width - margin, self.height - margin)
        panel_width = panel[2] - panel[0]
        left_gutter = max(round(70 * self.scale), round(panel_width * 0.055))
        right_gutter = max(round(24 * self.scale), round(panel_width * 0.028))
        header_height = max(
            round((48 if self.compact else 62) * self.scale),
            round(panel_height * (0.42 if self.compact else 0.28)),
        )
        footer_height = max(
            round(28 * self.scale),
            round(panel_height * 0.14),
            24 if self.compact else 20,
        )
        chart = (
            panel[0] + left_gutter,
            panel[1] + header_height,
            panel[2] - right_gutter,
            panel[3] - footer_height,
        )
        progress_height = max(3, round(4 * self.scale))
        progress = (
            chart[0],
            panel[3] - max(round(12 * self.scale), 8),
            chart[2],
            panel[3] - max(round(12 * self.scale), 8) + progress_height,
        )
        return OverlayLayout(
            panel=panel,
            chart=chart,
            progress=progress,
            margin=margin,
            radius=max(14, round(22 * self.scale)),
        )

    def _build_static_overlay(self) -> Image.Image:
        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # The frame stays dominant; only the lower field is darkened enough for
        # a readable curve, with no opaque or white plotting canvas.
        gradient_top = max(0, self.layout.panel[1] - round(self.height * 0.18))
        gradient_height = max(1, self.height - gradient_top)
        for row in range(gradient_height):
            amount = row / max(gradient_height - 1, 1)
            alpha = round((amount**1.55) * 142)
            draw.line((0, gradient_top + row, self.width, gradient_top + row), fill=(0, 5, 9, alpha))

        panel = self.layout.panel
        shadow = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow, "RGBA")
        offset = max(4, round(7 * self.scale))
        shadow_draw.rounded_rectangle(
            (panel[0], panel[1] + offset, panel[2], panel[3] + offset),
            radius=self.layout.radius,
            fill=(0, 0, 0, 130),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(max(5, round(11 * self.scale))))
        overlay = Image.alpha_composite(overlay, shadow)
        draw = ImageDraw.Draw(overlay, "RGBA")
        draw.rounded_rectangle(
            panel,
            radius=self.layout.radius,
            fill=self.theme.panel,
            outline=(224, 242, 248, 34),
            width=max(1, round(self.scale)),
        )

        chart = self.layout.chart
        for fraction in (0.0, 0.5, 1.0):
            y = round(_lerp(chart[3], chart[1], fraction))
            draw.line((chart[0], y, chart[2], y), fill=self.theme.grid, width=1)
            value = _lerp(self.y_min, self.y_max, fraction)
            label = f"{value:.2f}"
            label_width = _text_width(draw, label, self.font_tiny)
            draw.text(
                (chart[0] - max(8, round(10 * self.scale)) - label_width, y),
                label,
                font=self.font_tiny,
                fill=(*self.theme.muted, 205),
                anchor="lm",
            )

        # Static identity on the left: restrained typography and a live dot.
        dot_radius = max(3, round(4 * self.scale))
        header_y = panel[1] + max(18, round(22 * self.scale))
        dot_x = panel[0] + max(20, round(27 * self.scale))
        draw.ellipse(
            (dot_x - dot_radius, header_y - dot_radius, dot_x + dot_radius, header_y + dot_radius),
            fill=(*self.theme.accent, 255),
        )
        header_label = (
            # f"PREDICTED VALUE | EP {self.curve.episode_id:06d}"
            "PREDICTED VALUE"
            if self.compact
            else "PREDICTED VALUE"
        )
        draw.text(
            (dot_x + max(9, round(12 * self.scale)), header_y),
            header_label,
            font=self.font_small_bold,
            fill=(*self.theme.foreground, 245),
            anchor="lm",
        )
        if not self.compact:
            episode_text = f"EPISODE {self.curve.episode_id:06d}"
            draw.text(
                (
                    panel[0] + max(20, round(27 * self.scale)),
                    header_y + max(20, round(24 * self.scale)),
                ),
                episode_text,
                font=self.font_tiny,
                fill=(*self.theme.muted, 220),
                anchor="lm",
            )

        first_label = str(self.timeline_first)
        last_label = str(self.timeline_last)
        draw.text(
            (chart[0], self.layout.progress[1] - max(2, round(3 * self.scale))),
            first_label,
            font=self.font_tiny,
            fill=(*self.theme.muted, 185),
            anchor="lb",
        )
        draw.text(
            (chart[2], self.layout.progress[1] - max(2, round(3 * self.scale))),
            last_label,
            font=self.font_tiny,
            fill=(*self.theme.muted, 185),
            anchor="rb",
        )

        progress = self.layout.progress
        draw.rounded_rectangle(
            progress,
            radius=max(1, (progress[3] - progress[1]) // 2),
            fill=(225, 241, 247, 38),
        )

        # Small floating brand chip keeps the visual recognizable while leaving
        # the scene unobstructed.
        chip_text = self.title.upper()
        chip_padding_x = max(12, round(16 * self.scale))
        chip_padding_y = max(7, round(9 * self.scale))
        chip_width = _text_width(draw, chip_text, self.font_small_bold) + chip_padding_x * 2
        chip_box = (
            self.layout.margin,
            self.layout.margin,
            self.layout.margin + chip_width,
            self.layout.margin + chip_padding_y * 2 + max(12, round(15 * self.scale)),
        )
        draw.rounded_rectangle(
            chip_box,
            radius=max(9, round(13 * self.scale)),
            fill=(2, 9, 14, 145),
            outline=(235, 248, 252, 35),
            width=1,
        )
        draw.text(
            ((chip_box[0] + chip_box[2]) // 2, (chip_box[1] + chip_box[3]) // 2),
            chip_text,
            font=self.font_small_bold,
            fill=(*self.theme.foreground, 240),
            anchor="mm",
        )
        return overlay

    def x_for_frame(self, frame_index: int) -> int:
        chart = self.layout.chart
        amount = (frame_index - self.timeline_first) / (self.timeline_last - self.timeline_first)
        amount = min(1.0, max(0.0, amount))
        return round(_lerp(chart[0], chart[2], amount))

    def y_for_value(self, value: float) -> int:
        chart = self.layout.chart
        amount = (value - self.y_min) / (self.y_max - self.y_min)
        amount = min(1.0, max(0.0, amount))
        return round(_lerp(chart[3], chart[1], amount))

    def point_xy(self, frame_index: int, value: float) -> tuple[int, int]:
        return self.x_for_frame(frame_index), self.y_for_value(value)

    def _visible_segments(self, visible_count: int) -> list[list[tuple[int, int]]]:
        if visible_count < 1:
            return []
        segments: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        previous_frame: int | None = None
        for frame_index, value in zip(
            self.curve.frame_indices[:visible_count],
            self.curve.values[:visible_count],
            strict=True,
        ):
            if previous_frame is not None and frame_index != previous_frame + 1:
                if current:
                    segments.append(current)
                current = []
            current.append(self.point_xy(frame_index, value))
            previous_frame = frame_index
        if current:
            segments.append(current)
        return segments

    def _draw_curve(self, dynamic: Image.Image, state: CurveFrameState) -> None:
        chart = self.layout.chart
        chart_width = chart[2] - chart[0] + 1
        chart_height = chart[3] - chart[1] + 1
        segments = self._visible_segments(state.visible_count)
        if not segments:
            return

        curve_layer = Image.new("RGBA", (chart_width, chart_height), (0, 0, 0, 0))
        curve_draw = ImageDraw.Draw(curve_layer, "RGBA")
        local_segments = [
            [(x - chart[0], y - chart[1]) for x, y in segment] for segment in segments
        ]

        # A low-opacity area wash gives depth without turning the overlay into a
        # conventional chart card.
        for segment in local_segments:
            if len(segment) >= 2:
                polygon = [segment[0], *segment[1:], (segment[-1][0], chart_height), (segment[0][0], chart_height)]
                curve_draw.polygon(polygon, fill=(*self.theme.accent, 18))

        glow = Image.new("RGBA", curve_layer.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        glow_width = max(5, round(8 * self.scale))
        for segment in local_segments:
            if len(segment) >= 2:
                glow_draw.line(segment, fill=(*self.theme.accent, 170), width=glow_width, joint="curve")
            else:
                x, y = segment[0]
                glow_draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(*self.theme.accent, 170))
        glow = glow.filter(ImageFilter.GaussianBlur(max(3, round(7 * self.scale))))
        # Decorative blur must not visually spill into future timeline space.
        playhead_local_x = max(0, min(chart_width - 1, self.x_for_frame(state.frame_index) - chart[0]))
        clipped_glow = Image.new("RGBA", glow.size, (0, 0, 0, 0))
        clipped_glow.paste(glow.crop((0, 0, playhead_local_x + 1, chart_height)), (0, 0))
        glow = clipped_glow
        curve_layer = Image.alpha_composite(curve_layer, glow)
        curve_draw = ImageDraw.Draw(curve_layer, "RGBA")
        line_width = max(3, round(3.2 * self.scale))
        for segment in local_segments:
            if len(segment) >= 2:
                curve_draw.line(
                    segment,
                    fill=(*self.theme.accent, 255),
                    width=line_width,
                    joint="curve",
                )
            else:
                x, y = segment[0]
                radius = max(2, round(3 * self.scale))
                curve_draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=(*self.theme.accent, 255),
                )
        dynamic.alpha_composite(curve_layer, dest=(chart[0], chart[1]))

    def _draw_marker(
        self,
        dynamic: Image.Image,
        state: CurveFrameState,
    ) -> None:
        if state.exact_point_index is None:
            if state.status == "terminal_hold" and state.latest_point_index is not None:
                point_index = state.latest_point_index
                x, y = self.point_xy(
                    self.curve.frame_indices[point_index], self.curve.values[point_index]
                )
                draw = ImageDraw.Draw(dynamic, "RGBA")
                radius = max(5, round(7 * self.scale))
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=(2, 11, 17, 150),
                    outline=(*self.theme.accent, 175),
                    width=max(1, round(2 * self.scale)),
                )
            return
        point_index = state.exact_point_index
        x, y = self.point_xy(
            self.curve.frame_indices[point_index], self.curve.values[point_index]
        )
        halo = Image.new("RGBA", dynamic.size, (0, 0, 0, 0))
        halo_radius = max(8, round(13 * self.scale))
        halo_extent = halo_radius + max(5, round(9 * self.scale))
        halo = Image.new(
            "RGBA",
            (halo_extent * 2 + 1, halo_extent * 2 + 1),
            (0, 0, 0, 0),
        )
        halo_draw = ImageDraw.Draw(halo, "RGBA")
        center = halo_extent
        halo_draw.ellipse(
            (
                center - halo_radius,
                center - halo_radius,
                center + halo_radius,
                center + halo_radius,
            ),
            fill=(*self.theme.accent, 180),
        )
        halo = halo.filter(ImageFilter.GaussianBlur(max(4, round(8 * self.scale))))
        clipped_halo = Image.new("RGBA", halo.size, (0, 0, 0, 0))
        clipped_halo.paste(halo.crop((0, 0, halo_extent + 1, halo.height)), (0, 0))
        halo = clipped_halo
        dynamic.alpha_composite(halo, dest=(x - halo_extent, y - halo_extent))
        draw = ImageDraw.Draw(dynamic, "RGBA")
        outer = max(5, round(7 * self.scale))
        inner = max(2, round(3 * self.scale))
        draw.ellipse(
            (x - outer, y - outer, x + outer, y + outer),
            fill=(*self.theme.accent, 255),
            outline=(235, 252, 255, 230),
            width=max(1, round(2 * self.scale)),
        )
        draw.ellipse(
            (x - inner, y - inner, x + inner, y + inner),
            fill=(248, 254, 255, 255),
        )

    def render(
        self,
        frame: Image.Image,
        *,
        frame_index: int,
        ordinal: int,
        total_frames: int,
    ) -> Image.Image:
        if frame.size != (self.width, self.height):
            raise ValueError(
                f"Overlay expected {self.width}x{self.height}, got {frame.width}x{frame.height}."
            )
        state = self.curve.frame_state(frame_index)
        result = Image.alpha_composite(frame.convert("RGBA"), self._static_overlay)
        dynamic = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(dynamic, "RGBA")
        chart = self.layout.chart
        panel = self.layout.panel

        playhead_x = self.x_for_frame(frame_index)
        draw.line(
            (playhead_x, chart[1], playhead_x, chart[3]),
            fill=(*self.theme.accent, 125),
            width=max(1, round(1.4 * self.scale)),
        )
        triangle = max(4, round(6 * self.scale))
        draw.polygon(
            [
                (playhead_x, chart[3] + triangle),
                (playhead_x - triangle, chart[3]),
                (playhead_x + triangle, chart[3]),
            ],
            fill=(*self.theme.accent, 225),
        )

        self._draw_curve(dynamic, state)
        self._draw_marker(dynamic, state)
        draw = ImageDraw.Draw(dynamic, "RGBA")

        latest_value = None
        if state.status in {"estimated", "terminal_hold"} and state.latest_point_index is not None:
            latest_value = self.curve.values[state.latest_point_index]
        if state.status == "warming_up":
            status = "BUILDING HISTORY"
        elif state.status == "estimated":
            status = "LIVE ESTIMATE"
        elif state.status == "terminal_hold":
            status = "TERMINAL FRAME | LAST ESTIMATE HELD"
        else:
            status = "NO ESTIMATE AT THIS FRAME"

        header_y = panel[1] + max(18, round(22 * self.scale))
        right_x = panel[2] - max(20, round(28 * self.scale))
        value_text = "--" if latest_value is None else f"{latest_value:+.3f}"
        value_font = self.font_small_bold if self.ultra_compact else self.font_value
        if self.ultra_compact:
            value_text = f"V {value_text}"
        draw.text(
            (
                right_x,
                header_y if self.ultra_compact else header_y - max(4, round(5 * self.scale)),
            ),
            value_text,
            font=value_font,
            fill=(*self.theme.foreground, 255),
            anchor="rm" if self.ultra_compact else "ra",
        )
        frame_prefix = {
            "warming_up": "WARM-UP | ",
            "terminal_hold": "HELD | ",
            "no_estimate": "NO ESTIMATE | ",
        }.get(state.status, "")
        frame_text = f"{frame_prefix}FRAME {frame_index}  |  {ordinal + 1}/{total_frames}"
        if not self.ultra_compact:
            draw.text(
                (right_x, header_y + max(22, round(27 * self.scale))),
                frame_text,
                font=self.font_tiny,
                fill=(*self.theme.muted, 220),
                anchor="ra",
            )

        if not self.compact:
            status_x = panel[0] + max(160, round(190 * self.scale))
            max_status_width = max(80, right_x - status_x - round(150 * self.scale))
            status = _truncate_text(draw, status, self.font_tiny, max_status_width)
            draw.text(
                (status_x, header_y),
                status,
                font=self.font_tiny,
                fill=(*self.theme.accent, 225),
                anchor="lm",
            )
            if self.camera_key:
                camera_text = _truncate_text(
                    draw,
                    self.camera_key,
                    self.font_tiny,
                    max_status_width,
                )
                draw.text(
                    (status_x, header_y + max(20, round(24 * self.scale))),
                    camera_text,
                    font=self.font_tiny,
                    fill=(*self.theme.muted, 190),
                    anchor="lm",
                )

        progress = self.layout.progress
        progress_amount = (frame_index - self.timeline_first) / (
            self.timeline_last - self.timeline_first
        )
        progress_amount = min(1.0, max(0.0, progress_amount))
        filled_right = round(_lerp(progress[0], progress[2], progress_amount))
        if filled_right > progress[0]:
            draw.rounded_rectangle(
                (progress[0], progress[1], filled_right, progress[3]),
                radius=max(1, (progress[3] - progress[1]) // 2),
                fill=(*self.theme.accent, 235),
            )

        if self.debug_alignment:
            debug_text = (
                f"video_ordinal={ordinal} -> frame_index={frame_index} | "
                f"visible_points={state.visible_count} | exact={state.exact_point_index is not None}"
            )
            draw.text(
                (self.layout.margin, self.height - max(3, round(4 * self.scale))),
                debug_text,
                font=self.font_tiny,
                fill=(238, 249, 252, 220),
                anchor="lb",
                stroke_width=2,
                stroke_fill=(0, 0, 0, 170),
            )

        result = Image.alpha_composite(result, dynamic)
        return result.convert("RGB")


def build_tick_values(lower: float, upper: float, count: int = 3) -> Iterable[float]:
    if count < 2:
        raise ValueError("Tick count must be at least two.")
    for index in range(count):
        yield _lerp(lower, upper, index / (count - 1))
