from __future__ import annotations

"""Episode-wise value/return curves for offline evaluation.

The temporal dataset contains overlapping history windows.  A window of
``history_size + 1`` observations produces one value for every observation in
its history, but most of those values are repeated by neighbouring windows.
For an episode curve we therefore deliberately keep *only the last valid
history position* from each window.  That position has the complete history
available to the model and maps to one unique ``(episode_id, frame_index)``.

This module keeps the aggregation and plotting code independent from the
distributed evaluation loop.  The loop supplies one record per window and
this module sorts by the actual frame index, rejects duplicate keys, and writes
machine-readable artifacts before rendering PNGs.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Episode curve {name} must be finite, got {value!r}.")
    return result


def build_episode_curves(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build sorted, duplicate-free curves from one record per eval window.

    Each input record must contain ``episode_id``, ``frame_index``, ``value``,
    and the supervised ``return`` target.  The returned list is sorted by
    episode id and frame index.  We intentionally raise on duplicate keys
    instead of silently averaging or truncating overlapping windows: a
    duplicate means the sampler/window alignment contract was violated and a
    plot would otherwise look plausible while being wrong.
    """

    normalized: list[tuple[int, int, float, float]] = []
    seen: set[tuple[int, int]] = set()
    for record in records:
        try:
            episode_id = int(record["episode_id"])
            frame_index = int(record["frame_index"])
            value = _finite(record["value"], name="value")
            target = _finite(record["return"], name="return")
        except KeyError as exc:
            raise KeyError(f"Episode curve record is missing {exc.args[0]!r}.") from exc
        key = (episode_id, frame_index)
        if key in seen:
            raise ValueError(
                "Duplicate episode/frame in evaluation curve records: "
                f"episode_id={episode_id}, frame_index={frame_index}. "
                "Each temporal window must contribute only its last valid timestep."
            )
        seen.add(key)
        normalized.append((episode_id, frame_index, value, target))

    normalized.sort(key=lambda item: (item[0], item[1]))
    curves: list[dict[str, Any]] = []
    for episode_id, frame_index, value, target in normalized:
        if not curves or curves[-1]["episode_id"] != episode_id:
            curves.append(
                {
                    "episode_id": episode_id,
                    "frame_indices": [],
                    "values": [],
                    "returns": [],
                }
            )
        curve = curves[-1]
        curve["frame_indices"].append(frame_index)
        curve["values"].append(value)
        curve["returns"].append(target)

    for curve in curves:
        values = curve["values"]
        targets = curve["returns"]
        errors = [value - target for value, target in zip(values, targets, strict=True)]
        count = len(errors)
        mse = sum(error * error for error in errors) / count if count else math.nan
        mae = sum(abs(error) for error in errors) / count if count else math.nan
        if count < 2:
            pearson = math.nan
        else:
            mean_value = sum(values) / count
            mean_target = sum(targets) / count
            numerator = sum(
                (value - mean_value) * (target - mean_target)
                for value, target in zip(values, targets, strict=True)
            )
            denom_value = sum((value - mean_value) ** 2 for value in values)
            denom_target = sum((target - mean_target) ** 2 for target in targets)
            denominator = math.sqrt(denom_value * denom_target)
            pearson = numerator / denominator if denominator > 0 else math.nan
        curve["metrics"] = {
            "count": count,
            "value_mse": mse,
            "value_rmse": math.sqrt(mse) if math.isfinite(mse) else math.nan,
            "value_mae": mae,
            "value_pearson": pearson,
        }
    return curves


def _json_safe(value: Any) -> Any:
    """Convert NaN metrics to JSON ``null`` while preserving finite numbers."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def write_episode_curve_artifacts(
    curves: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    render_plots: bool = True,
) -> dict[str, Any]:
    """Write JSON/CSV and one value-vs-return PNG per episode.

    Plotting is intentionally lazy-imported so training and metric-only
    evaluation do not require a GUI backend.  ``matplotlib`` is switched to
    the non-interactive ``Agg`` backend before importing pyplot.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "episode_curves.json"
    metrics_path = output_dir / "episode_metrics.json"
    summary_path = output_dir / "episode_curves_summary.json"
    csv_path = output_dir / "episode_curves.csv"
    json_path.write_text(
        json.dumps(_json_safe(curves), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(
            _json_safe(
                [{"episode_id": curve["episode_id"], **curve["metrics"]} for curve in curves]
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_id", "frame_index", "value", "return"])
        for curve in curves:
            for frame_index, value, target in zip(
                curve["frame_indices"], curve["values"], curve["returns"], strict=True
            ):
                writer.writerow([curve["episode_id"], frame_index, value, target])

    plot_paths: list[str] = []
    if render_plots and curves:
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "Episode curve rendering requires matplotlib. Install the project dependencies "
                "or rerun evaluation with --no-episode-curves."
            ) from exc

        for curve in curves:
            episode_id = curve["episode_id"]
            frame_indices = curve["frame_indices"]
            values = curve["values"]
            targets = curve["returns"]
            metrics = curve["metrics"]
            figure, axis = plt.subplots(figsize=(12, 5.5))
            marker = "o" if len(frame_indices) == 1 else None
            axis.plot(
                frame_indices,
                targets,
                label="Return target",
                color="#2563eb",
                linewidth=2,
                marker=marker,
            )
            axis.plot(
                frame_indices,
                values,
                label="Predicted value",
                color="#dc2626",
                linestyle="--",
                linewidth=2,
                marker=marker,
            )
            axis.set_title(f"Episode {episode_id}: value vs return")
            axis.set_xlabel("frame_index")
            axis.set_ylabel("value / return")
            axis.grid(True, alpha=0.25)
            axis.legend()
            metric_text = (
                f"n={metrics['count']}\n"
                f"MSE={metrics['value_mse']:.5g}\n"
                f"MAE={metrics['value_mae']:.5g}"
            )
            axis.text(
                0.01,
                0.98,
                metric_text,
                transform=axis.transAxes,
                va="top",
                ha="left",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            figure.tight_layout()
            path = output_dir / f"episode-{episode_id}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            plot_paths.append(str(path.resolve()))

    point_count = sum(len(curve["frame_indices"]) for curve in curves)
    summary = {
        "enabled": bool(render_plots),
        "num_episodes": len(curves),
        "num_points": point_count,
        "directory": str(output_dir.resolve()),
        "json": str(json_path.resolve()),
        "episode_metrics": str(metrics_path.resolve()),
        "csv": str(csv_path.resolve()),
        "plot_count": len(plot_paths),
        "plots": plot_paths,
    }
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary["summary"] = str(summary_path.resolve())
    return summary
