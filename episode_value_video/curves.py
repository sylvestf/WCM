from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


class CurveDataError(ValueError):
    """Raised when an evaluator curve artifact cannot be aligned safely."""


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise CurveDataError(f"{field} must be an integer, got {value!r}.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CurveDataError(f"{field} must be an integer, got {value!r}.") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(result)
    if not math.isfinite(numeric) or numeric != result:
        raise CurveDataError(f"{field} must be an exact finite integer, got {value!r}.")
    return result


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CurveDataError(f"{field} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(result):
        raise CurveDataError(f"{field} must be finite, got {value!r}.")
    return result


@dataclass(frozen=True, slots=True)
class CurveFrameState:
    """The value-curve state that is legal to show on one video frame."""

    frame_index: int
    visible_count: int
    exact_point_index: int | None
    latest_point_index: int | None
    status: str


@dataclass(frozen=True, slots=True)
class EpisodeCurve:
    episode_id: int
    frame_indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.frame_indices:
            raise CurveDataError(f"episode_id={self.episode_id} has no predicted value points.")
        if len(self.frame_indices) != len(self.values):
            raise CurveDataError(
                f"episode_id={self.episode_id} has {len(self.frame_indices)} frame indices "
                f"but {len(self.values)} values."
            )
        previous: int | None = None
        for position, (frame_index, value) in enumerate(
            zip(self.frame_indices, self.values, strict=True)
        ):
            if previous is not None and frame_index <= previous:
                kind = "duplicate" if frame_index == previous else "unsorted"
                raise CurveDataError(
                    f"episode_id={self.episode_id} contains a {kind} frame_index={frame_index} "
                    f"at point {position}."
                )
            if not math.isfinite(value):
                raise CurveDataError(
                    f"episode_id={self.episode_id} contains a non-finite value at "
                    f"frame_index={frame_index}."
                )
            previous = frame_index

    @property
    def first_frame(self) -> int:
        return self.frame_indices[0]

    @property
    def last_frame(self) -> int:
        return self.frame_indices[-1]

    def frame_state(self, frame_index: int) -> CurveFrameState:
        """Return the reveal state without interpolating or exposing future points."""

        visible_count = bisect.bisect_right(self.frame_indices, frame_index)
        exact_position = bisect.bisect_left(self.frame_indices, frame_index)
        exact = (
            exact_position
            if exact_position < len(self.frame_indices)
            and self.frame_indices[exact_position] == frame_index
            else None
        )
        latest = visible_count - 1 if visible_count else None
        if frame_index < self.first_frame:
            status = "warming_up"
        elif exact is not None:
            status = "estimated"
        elif frame_index > self.last_frame:
            status = "terminal_hold"
        else:
            status = "no_estimate"
        return CurveFrameState(
            frame_index=frame_index,
            visible_count=visible_count,
            exact_point_index=exact,
            latest_point_index=latest,
            status=status,
        )

    def value_at(self, frame_index: int) -> float | None:
        state = self.frame_state(frame_index)
        if state.exact_point_index is None:
            return None
        return self.values[state.exact_point_index]


def _payload_entries(payload: Any, *, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        candidate = payload.get("curves", payload.get("episode_curves"))
        if not isinstance(candidate, list):
            raise CurveDataError(
                f"{source} must contain the evaluator's curve list, or an object with a "
                "'curves'/'episode_curves' list."
            )
        entries = candidate
    else:
        raise CurveDataError(f"{source} must contain a JSON list of episode curves.")
    if not all(isinstance(entry, dict) for entry in entries):
        raise CurveDataError(f"{source} contains a non-object episode curve entry.")
    return entries


def load_episode_curves(path: str | Path) -> list[EpisodeCurve]:
    """Load predicted values from ``episode_curves.json``.

    The evaluator's ``returns`` and ``metrics`` fields are deliberately ignored.
    Only the model prediction and its exact frame key enter the video renderer.
    """

    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Episode curve artifact does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise CurveDataError(f"Invalid JSON in {source}: {exc}") from exc

    curves: list[EpisodeCurve] = []
    seen_episodes: set[int] = set()
    for entry_position, entry in enumerate(_payload_entries(payload, source=source)):
        missing = {"episode_id", "frame_indices", "values"} - set(entry)
        if missing:
            raise CurveDataError(
                f"Curve entry {entry_position} in {source} is missing fields: {sorted(missing)}"
            )
        episode_id = _strict_int(entry["episode_id"], field="episode_id")
        if episode_id in seen_episodes:
            raise CurveDataError(f"{source} contains duplicate episode_id={episode_id} entries.")
        frame_values = entry["frame_indices"]
        prediction_values = entry["values"]
        if not isinstance(frame_values, Sequence) or isinstance(frame_values, (str, bytes)):
            raise CurveDataError(f"episode_id={episode_id} frame_indices must be a sequence.")
        if not isinstance(prediction_values, Sequence) or isinstance(
            prediction_values, (str, bytes)
        ):
            raise CurveDataError(f"episode_id={episode_id} values must be a sequence.")
        if len(frame_values) != len(prediction_values):
            raise CurveDataError(
                f"episode_id={episode_id} has {len(frame_values)} frame indices but "
                f"{len(prediction_values)} values."
            )

        pairs = [
            (
                _strict_int(frame_index, field=f"episode {episode_id} frame_index"),
                _finite_float(value, field=f"episode {episode_id} value"),
            )
            for frame_index, value in zip(frame_values, prediction_values, strict=True)
        ]
        pairs.sort(key=lambda item: item[0])
        for left, right in zip(pairs, pairs[1:]):
            if left[0] == right[0]:
                raise CurveDataError(
                    f"episode_id={episode_id} contains duplicate frame_index={left[0]}."
                )
        curve = EpisodeCurve(
            episode_id=episode_id,
            frame_indices=tuple(frame for frame, _ in pairs),
            values=tuple(value for _, value in pairs),
        )
        curves.append(curve)
        seen_episodes.add(episode_id)

    curves.sort(key=lambda curve: curve.episode_id)
    if not curves:
        raise CurveDataError(f"{source} contains no episode curves.")
    return curves


def select_curves(
    curves: Iterable[EpisodeCurve],
    episode_ids: Iterable[int] | None = None,
    max_episodes: int | None = None,
) -> list[EpisodeCurve]:
    available = {curve.episode_id: curve for curve in curves}
    if episode_ids is None:
        selected = [available[key] for key in sorted(available)]
    else:
        requested = list(dict.fromkeys(int(value) for value in episode_ids))
        missing = sorted(set(requested) - set(available))
        if missing:
            raise CurveDataError(
                f"Requested episodes are absent from the curve artifact: {missing[:20]}"
            )
        selected = [available[episode_id] for episode_id in requested]
    if max_episodes is not None:
        if max_episodes < 1:
            raise CurveDataError("max_episodes must be positive.")
        selected = selected[:max_episodes]
    return selected


def value_domain(
    curves: Iterable[EpisodeCurve],
    *,
    requested_min: float | None = None,
    requested_max: float | None = None,
) -> tuple[float, float]:
    values = [value for curve in curves for value in curve.values]
    if not values:
        raise CurveDataError("Cannot compute a value domain without prediction points.")
    data_lower, data_upper = min(values), max(values)
    raw_span = data_upper - data_lower
    magnitude = max(abs(data_lower), abs(data_upper), 1.0)
    span = max(raw_span, magnitude * 0.08, 0.08)
    padding = span * 0.14
    lower = data_lower - padding if requested_min is None else _finite_float(requested_min, field="y_min")
    upper = data_upper + padding if requested_max is None else _finite_float(requested_max, field="y_max")
    if not lower < upper:
        if requested_min is not None and requested_max is not None:
            raise CurveDataError(f"y_min must be smaller than y_max, got {lower} >= {upper}.")
        if requested_min is not None:
            upper = lower + span
        else:
            lower = upper - span
    return lower, upper
