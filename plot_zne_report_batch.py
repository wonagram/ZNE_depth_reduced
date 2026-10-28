"""Create one method-by-noise ZNE summary figure per experiment case.

For each fixed (num_qubits, unitary, depth, platform) setting, this script scans
``zne_reports`` and combines the available reports into one figure:

* x-axis: constant depolarizing (0.2%), increasing Pauli, increasing Pauli + AD;
* color: circuit folding, global width folding, local width folding;
* marker: exponential or linear extrapolation;
* gate-count definition: before-transpilation weighted count only.

A complete per-circuit group contains 3 methods x 3 noise models = 9 JSON reports.
Incomplete per-circuit groups are drawn automatically when at least one method
has results for at least two target noise models.  This allows the current partial
0.2% data set to be visualized while local-folding reports are still pending.

In addition, the script creates two across-circuit statistical summaries for
each (num_qubits, depth round, platform) group when all four circuit families
(random_type, HEA-CZ, HEA-CRZ, HEA-RZZ) are complete and at least two folding
methods are complete across all three target noise models. If all three methods
are complete, all three are included; otherwise the two complete methods are used:

* a bar chart of mean absolute ZNE error with SD error bars;
* a box plot with the four actual circuit-family errors overlaid.

The across-circuit summaries intentionally exclude 1q, because only the random
circuit family is defined for 1q in this experiment and therefore there is no
four-circuit average to report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_gate_axis_zne as gate_plotter

try:
    import experiment_config
except ImportError:  # fallback: infer ordinal depth rounds from available reports
    experiment_config = None


# ============================================================
# User configuration
# ============================================================
REPORT_DIRECTORY = "./zne_reports"
PLOT_OUTPUT_ROOT = "./zne_plots_batch_0p2pct"
ALLOW_PARTIAL_GROUPS = True
SAVE_DPI = 200
SHOW_PLOTS = False

NUM_QUBITS_FILTER = None       # Example: [2, 3, 5]
UNITARY_FILTER = None          # Example: ["hea_cz", "hea_crz"]
DEPTH_FILTER = None            # Example: [30, 32]
PLATFORM_FILTER = None         # Example: ["GPU"]

METHOD_ORDER = ("circuit_folding", "global", "local")

# Aggregate analysis across the four circuit families.  A circuit-family average
# is formed only after computing the absolute ZNE error for each circuit.
CIRCUIT_ORDER = ("random_type", "hea_cz", "hea_crz", "hea_rzz")
CIRCUIT_LABELS = {
    "random_type": "Random",
    "hea_cz": "HEA-CZ",
    "hea_crz": "HEA-CRZ",
    "hea_rzz": "HEA-RZZ",
}
GENERATE_CIRCUIT_AGGREGATE_PLOTS = True
AGGREGATE_STD_DDOF = 0  # descriptive SD across the four selected circuit families
AGGREGATE_BAR_FIGSIZE = (13, 6.5)
AGGREGATE_BOX_FIGSIZE = (13, 6.5)
AGGREGATE_CLUSTER_SPAN = 0.72
AGGREGATE_BAR_WIDTH = 0.105

# Extra visualization for pathological ZNE extrapolation outliers.
# The ordinary mean±SD bar plot is ALWAYS kept. If any individual absolute
# ZNE error included in an aggregate is >= this threshold, an additional
# truncated/broken-bar figure is generated so the remaining bars stay readable.
BROKEN_BAR_ERROR_THRESHOLD = 4.0
BROKEN_BAR_HEADROOM = 1.22
BROKEN_BAR_MIN_YMAX = 0.05

NOISE_PROFILES = {
    # IMPORTANT: the historical 2% ``constant_depolarizing`` reports are
    # intentionally NOT included.  The constant-DP point in every new figure
    # comes only from ``small_constant_depolarizing_0p2pct`` reports.
    "small_constant_depolarizing_0p2pct": {
        "label": "Constant depolarizing\n(0.2%)",
    },
    "increasing_pauli": {
        "label": "Increasing\nPauli",
    },
    "increasing_pauli_amplitude_damping": {
        "label": "Increasing Pauli +\namplitude damping",
    },
}


@dataclass(frozen=True)
class ReportRecord:
    path: Path
    report: dict[str, Any]
    num_qubits: int
    unitary: str
    depth: int
    method: str
    execution_tag: str
    platform_tag: str
    noise_key: str

    @property
    def case_key(self) -> tuple[int, str, int, str]:
        return self.num_qubits, self.unitary, self.depth, self.platform_tag


def _safe_tag(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("_") or "unknown"


def _execution_tag_from_filename(path: Path, report: dict[str, Any]) -> str:
    num_qubits = int(report["num_qubits"])
    unitary = str(report["unitary_type"])
    depth = int(report["depth"])
    method = str(report["method"])
    prefix = f"{num_qubits}q_{unitary}_{depth}_"
    suffix = f"_{method}.json"
    name = path.name
    if name.startswith(prefix) and name.endswith(suffix):
        middle = name[len(prefix):-len(suffix)]
        if middle:
            return middle
    return gate_plotter._backend_tag(report)


def _split_noise_execution_tag(execution_tag: str) -> tuple[str, str] | None:
    for noise_key in sorted(NOISE_PROFILES, key=len, reverse=True):
        suffix = f"_{noise_key}"
        if execution_tag.endswith(suffix):
            platform_tag = execution_tag[:-len(suffix)]
            if platform_tag:
                return platform_tag, noise_key
    return None


def _passes_filters(record: ReportRecord) -> bool:
    if NUM_QUBITS_FILTER is not None and record.num_qubits not in NUM_QUBITS_FILTER:
        return False
    if UNITARY_FILTER is not None and record.unitary not in UNITARY_FILTER:
        return False
    if DEPTH_FILTER is not None and record.depth not in DEPTH_FILTER:
        return False
    if PLATFORM_FILTER is not None and record.platform_tag not in PLATFORM_FILTER:
        return False
    return True


def _load_records(report_directory: Path):
    records: list[ReportRecord] = []
    unreadable: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []

    for path in sorted(report_directory.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            execution_tag = _execution_tag_from_filename(path, report)
            split = _split_noise_execution_tag(execution_tag)
            if split is None:
                ignored.append({
                    "path": str(path),
                    "reason": "execution tag is not one of the three target noise profiles (0.2% constant DP, increasing Pauli, increasing Pauli + AD)",
                })
                continue
            platform_tag, noise_key = split
            record = ReportRecord(
                path=path.resolve(),
                report=report,
                num_qubits=int(report["num_qubits"]),
                unitary=str(report["unitary_type"]),
                depth=int(report["depth"]),
                method=str(report["method"]),
                execution_tag=execution_tag,
                platform_tag=platform_tag,
                noise_key=noise_key,
            )
        except Exception as error:
            unreadable.append({"path": str(path), "error": repr(error)})
            continue

        if _passes_filters(record):
            records.append(record)

    return records, unreadable, ignored


def _ideal_lookup(records: list[ReportRecord]):
    candidates: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for record in records:
        ideal = record.report.get("ideal_expectation")
        if ideal is not None:
            candidates[(record.num_qubits, record.unitary, record.depth)].append(
                float(ideal)
            )

    result: dict[tuple[int, str, int], float] = {}
    for key, values in candidates.items():
        reference = values[0]
        if any(abs(value - reference) > 1e-9 for value in values[1:]):
            raise ValueError(f"Inconsistent ideal_expectation values for {key}")
        result[key] = reference
    return result


def _build_groups(records: list[ReportRecord]):
    groups: dict[
        tuple[int, str, int, str],
        dict[tuple[str, str], ReportRecord],
    ] = defaultdict(dict)
    duplicates: list[dict[str, Any]] = []

    for record in records:
        combination = (record.method, record.noise_key)
        if combination in groups[record.case_key]:
            duplicates.append({
                "case": record.case_key,
                "combination": combination,
                "paths": [
                    str(groups[record.case_key][combination].path),
                    str(record.path),
                ],
            })
        else:
            groups[record.case_key][combination] = record
    return groups, duplicates



def _depth_round_lookup(records: list[ReportRecord]) -> dict[tuple[int, str, int], int]:
    """Map (num_qubits, unitary, depth) to the experiment-config ordinal round.

    Different circuit families intentionally use different physical depths in the
    same round (for example 3q: random=27, CRZ=30, CZ/RZZ=32).  Therefore the
    aggregate analysis must align circuits by round index, not by raw depth.
    """
    mapping: dict[tuple[int, str, int], int] = {}

    if experiment_config is not None:
        for group in getattr(experiment_config, "EXPERIMENTS", []):
            unitary = str(group.get("unitary", ""))
            if unitary not in CIRCUIT_ORDER:
                continue
            for raw_nq, depths in group.get("depths_by_num_qubits", {}).items():
                nq = int(raw_nq)
                for round_index, depth in enumerate(depths, start=1):
                    key = (nq, unitary, int(depth))
                    previous = mapping.get(key)
                    if previous is not None and previous != round_index:
                        raise ValueError(
                            f"Conflicting depth-round mapping for {key}: "
                            f"{previous} vs {round_index}"
                        )
                    mapping[key] = round_index

    # Fallback for any report not covered by the current configuration: use the
    # ordinal position of its depth within that (nq, unitary) family.
    depths_by_family: dict[tuple[int, str], set[int]] = defaultdict(set)
    for record in records:
        if record.unitary in CIRCUIT_ORDER:
            depths_by_family[(record.num_qubits, record.unitary)].add(record.depth)
    for (nq, unitary), depths in depths_by_family.items():
        for round_index, depth in enumerate(sorted(depths), start=1):
            mapping.setdefault((nq, unitary, depth), round_index)
    return mapping


def _build_circuit_aggregate_groups(
    records: list[ReportRecord],
    round_lookup: dict[tuple[int, str, int], int],
):
    groups: dict[
        tuple[int, int, str],
        dict[str, dict[tuple[str, str], ReportRecord]],
    ] = defaultdict(lambda: defaultdict(dict))
    unassigned: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []

    for record in records:
        if record.unitary not in CIRCUIT_ORDER:
            continue
        round_index = round_lookup.get(
            (record.num_qubits, record.unitary, record.depth)
        )
        if round_index is None:
            unassigned.append({
                "path": str(record.path),
                "reason": "could not map report depth to an experiment round",
            })
            continue
        group_key = (record.num_qubits, round_index, record.platform_tag)
        combination = (record.method, record.noise_key)
        circuit_map = groups[group_key][record.unitary]
        if combination in circuit_map:
            duplicates.append({
                "group": group_key,
                "unitary": record.unitary,
                "combination": combination,
                "paths": [str(circuit_map[combination].path), str(record.path)],
            })
            continue
        circuit_map[combination] = record
    return groups, unassigned, duplicates


def _darken_color(color: str, factor: float = 0.62):
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


def _aggregate_error_values(
    circuit_records: dict[str, dict[tuple[str, str], ReportRecord]],
    ideal_values: dict[tuple[int, str, int], float],
):
    """Return 4-circuit absolute-error vectors for methods complete across all noises.

    A method is included in the aggregate only when that same method has all three
    target noise profiles for all four circuit families.  This makes partial
    aggregate plots possible without mixing incomparable method coverage.
    """
    missing_fits: list[dict[str, Any]] = []
    missing_ideals: list[dict[str, Any]] = []
    incomplete_methods: dict[str, list[dict[str, Any]]] = {}

    # All four circuit families are required for an across-circuit statistic.
    missing_circuits = [
        unitary for unitary in CIRCUIT_ORDER if unitary not in circuit_records
    ]
    if missing_circuits:
        return None, {
            "missing_circuits": missing_circuits,
            "incomplete_methods": incomplete_methods,
            "missing_fits": missing_fits,
            "missing_ideals": missing_ideals,
        }, ()

    aggregate_methods: list[str] = []
    for method in METHOD_ORDER:
        method_missing: list[dict[str, Any]] = []
        for unitary in CIRCUIT_ORDER:
            available = circuit_records.get(unitary, {})
            missing_noise = [
                noise for noise in NOISE_PROFILES
                if (method, noise) not in available
            ]
            if missing_noise:
                method_missing.append({
                    "unitary": unitary,
                    "missing_noise": missing_noise,
                })
        if method_missing:
            incomplete_methods[method] = method_missing
        else:
            aggregate_methods.append(method)

    # Do not create an across-circuit aggregate from only one folding method.
    # Use all three methods when all are complete; while jobs are still running,
    # allow exactly the two methods that are complete across all 3 target noises
    # and all 4 circuit families.
    if len(aggregate_methods) < 2:
        return None, {
            "missing_circuits": [],
            "incomplete_methods": incomplete_methods,
            "complete_methods": list(aggregate_methods),
            "minimum_methods_required": 2,
            "missing_fits": missing_fits,
            "missing_ideals": missing_ideals,
        }, tuple(aggregate_methods)

    vectors: dict[tuple[str, str, str], list[float]] = {}
    for noise in NOISE_PROFILES:
        for method in aggregate_methods:
            for fit_key in ("exponential", "linear"):
                values: list[float] = []
                for unitary in CIRCUIT_ORDER:
                    record = circuit_records[unitary][(method, noise)]
                    ideal_key = (record.num_qubits, record.unitary, record.depth)
                    ideal = ideal_values.get(ideal_key)
                    if ideal is None:
                        missing_ideals.append({
                            "unitary": unitary,
                            "method": method,
                            "noise": noise,
                            "depth": record.depth,
                        })
                        continue
                    fit = record.report["cases"][gate_plotter.PRIMARY_CASE_KEY][fit_key]
                    if not fit.get("success") or fit.get("at_N_zero") is None:
                        missing_fits.append({
                            "unitary": unitary,
                            "method": method,
                            "noise": noise,
                            "fit": fit_key,
                            "depth": record.depth,
                        })
                        continue
                    error = abs(float(fit["at_N_zero"]) - float(ideal))
                    values.append(error)
                vectors[(noise, method, fit_key)] = values

    problems = {
        "missing_circuits": [],
        "incomplete_methods": incomplete_methods,
        "complete_methods": list(aggregate_methods),
        "minimum_methods_required": 2,
        "missing_fits": missing_fits,
        "missing_ideals": missing_ideals,
    }
    if missing_fits or missing_ideals:
        return None, problems, tuple(aggregate_methods)
    if any(len(values) != len(CIRCUIT_ORDER) for values in vectors.values()):
        return None, problems, tuple(aggregate_methods)
    return vectors, problems, tuple(aggregate_methods)

def _series_order(methods: tuple[str, ...] | list[str]):
    return [
        (method, fit_key, fit_label, marker)
        for method in methods
        for fit_key, fit_label, marker in (
            ("exponential", "Exponential", gate_plotter.EXPONENTIAL_ZERO_MARKER),
            ("linear", "Linear", gate_plotter.LINEAR_ZERO_MARKER),
        )
    ]

def _aggregate_depth_summary(
    circuit_records: dict[str, dict[tuple[str, str], ReportRecord]]
) -> str:
    parts = []
    for unitary in CIRCUIT_ORDER:
        records = circuit_records.get(unitary, {})
        representative = next(iter(records.values()), None)
        if representative is not None:
            parts.append(f"{CIRCUIT_LABELS[unitary]}={representative.depth}")
    return ", ".join(parts)


def _save_legend_only(
    handles,
    labels,
    output_path: Path,
    *,
    ncol: int = 2,
):
    """Save a tightly cropped legend-only PNG with the original legend sizing."""
    if not handles:
        return None
    legend_figure = plt.figure(figsize=(7.0, 4.0))
    legend_artist = legend_figure.legend(
        handles, labels, loc="center", frameon=True, ncol=ncol
    )
    legend_figure.canvas.draw()
    renderer = legend_figure.canvas.get_renderer()
    bbox = legend_artist.get_window_extent(renderer=renderer)
    bbox_inches = bbox.transformed(legend_figure.dpi_scale_trans.inverted())
    legend_figure.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches=bbox_inches,
        pad_inches=0.12,
    )
    plt.close(legend_figure)
    return output_path


def _artist_display_bbox(artist, renderer, axis):
    """Best-effort display-space bounding box for a plotted data artist."""
    try:
        bbox = artist.get_window_extent(renderer=renderer)
        values = np.asarray([bbox.x0, bbox.y0, bbox.x1, bbox.y1], dtype=float)
        if np.isfinite(values).all() and bbox.width > 0 and bbox.height > 0:
            return bbox
    except Exception:
        pass

    # Scatter collections often do not expose a useful window extent.
    try:
        offsets = np.asarray(artist.get_offsets(), dtype=float)
        if offsets.ndim == 2 and offsets.shape[0] and offsets.shape[1] >= 2:
            finite = np.isfinite(offsets[:, 0]) & np.isfinite(offsets[:, 1])
            offsets = offsets[finite]
            if offsets.size:
                disp = axis.transData.transform(offsets[:, :2])
                sizes = np.asarray(artist.get_sizes(), dtype=float)
                max_size = float(np.nanmax(sizes)) if sizes.size else 36.0
                radius_pt = max(max_size, 1.0) ** 0.5 / 2.0
                radius_px = radius_pt * axis.figure.dpi / 72.0 + 2.0
                from matplotlib.transforms import Bbox
                return Bbox.from_extents(
                    float(np.min(disp[:, 0]) - radius_px),
                    float(np.min(disp[:, 1]) - radius_px),
                    float(np.max(disp[:, 0]) + radius_px),
                    float(np.max(disp[:, 1]) + radius_px),
                )
    except Exception:
        pass
    return None


def _legend_overlaps_data(legend, data_artists, figure, axis) -> bool:
    """Return True if the legend touches any explicitly tracked data artist."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer)
    for artist in data_artists:
        if artist is None or not getattr(artist, "get_visible", lambda: True)():
            continue
        bbox = _artist_display_bbox(artist, renderer, axis)
        if bbox is not None and legend_bbox.overlaps(bbox):
            return True
    return False


def _save_plot_threeways(
    figure,
    axis,
    *,
    handles,
    output_path: Path,
    ncol: int = 2,
):
    """Always save with-legend, no-legend, and legend-only variants."""
    labels = [handle.get_label() for handle in handles]
    legend_artist = None
    if handles:
        legend_artist = axis.legend(handles=handles, loc="best", ncol=ncol)

    figure.tight_layout()

    no_legend_path = output_path.with_name(output_path.stem + "_no_legend.png")
    legend_path = output_path.with_name(output_path.stem + "_legend.png")

    # 1) Main plot with legend.
    figure.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")

    # 2) Same plot without legend.
    if legend_artist is not None:
        legend_artist.remove()
    figure.savefig(no_legend_path, dpi=SAVE_DPI, bbox_inches="tight")

    # 3) Legend only, using the exact same handles/labels/default typography.
    if handles:
        _save_legend_only(handles, labels, legend_path, ncol=ncol)

    return no_legend_path, legend_path if handles else None

def _plot_across_circuit_bar(
    vectors: dict[tuple[str, str, str], list[float]],
    *,
    methods: tuple[str, ...],
    output_path: Path,
    title_prefix: str,
    depth_summary: str,
):
    figure, axis = plt.subplots(figsize=AGGREGATE_BAR_FIGSIZE)
    noises = list(NOISE_PROFILES)
    x_base = np.arange(len(noises), dtype=float)
    series = _series_order(methods)
    offsets = np.linspace(
        -AGGREGATE_CLUSTER_SPAN / 2.0,
        AGGREGATE_CLUSTER_SPAN / 2.0,
        len(series),
    )

    legend_handles = []
    data_artists = []
    for index, (method, fit_key, fit_label, _marker) in enumerate(series):
        style = gate_plotter._method_style(method)
        facecolor = style["color"]
        edgecolor = "black" if fit_key == "linear" else _darken_color(facecolor)
        means = []
        stds = []
        for noise in noises:
            values = np.asarray(vectors[(noise, method, fit_key)], dtype=float)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values, ddof=AGGREGATE_STD_DDOF)))

        bar_container = axis.bar(
            x_base + offsets[index],
            means,
            width=AGGREGATE_BAR_WIDTH,
            yerr=stds,
            capsize=3,
            color=facecolor,
            edgecolor=edgecolor,
            linewidth=1.8,
            alpha=0.86,
            error_kw={"elinewidth": 1.1, "capthick": 1.1},
            zorder=3,
        )
        data_artists.extend(list(bar_container.patches))
        if getattr(bar_container, "errorbar", None) is not None:
            errorbar = bar_container.errorbar
            data_line, caplines, barlinecols = errorbar.lines
            if data_line is not None:
                data_artists.append(data_line)
            data_artists.extend(list(caplines))
            data_artists.extend(list(barlinecols))
        legend_handles.append(
            Patch(
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.8,
                label=f"{style['label']} {fit_label}",
            )
        )

    axis.set_xticks(x_base)
    axis.set_xticklabels([NOISE_PROFILES[key]["label"] for key in noises])
    axis.set_ylabel(
        r"Mean absolute ZNE error $|E_{\mathrm{ZNE}}-E_{\mathrm{ideal}}|$"
        "\n(error bar: SD across 4 circuit types)"
    )
    axis.set_xlabel("Noise model")
    axis.set_title(
        f"{title_prefix}\nMean ± SD across Random, HEA-CZ, HEA-CRZ, and HEA-RZZ\n"
        f"Depths in this round: {depth_summary}",
        fontweight="bold",
    )
    axis.grid(True, axis="y", alpha=0.3, zorder=0)
    axis.set_xlim(-0.55, len(noises) - 0.45)
    figure.tight_layout()
    _save_plot_threeways(
        figure,
        axis,
        handles=legend_handles,
        output_path=output_path,
        ncol=2,
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)



def _aggregate_max_individual_error(
    vectors: dict[tuple[str, str, str], list[float]],
) -> float:
    """Largest finite individual circuit-family absolute error in an aggregate."""
    finite_values: list[float] = []
    for values in vectors.values():
        array = np.asarray(values, dtype=float)
        finite_values.extend(array[np.isfinite(array)].tolist())
    return float(max(finite_values)) if finite_values else float("nan")


def _broken_bar_display_limits(
    vectors: dict[tuple[str, str, str], list[float]],
    methods: tuple[str, ...],
) -> tuple[float, float]:
    """Choose a compact y-range using only non-pathological bar series."""
    normal_upper: list[float] = []
    normal_lower: list[float] = []
    for noise in NOISE_PROFILES:
        for method in methods:
            for fit_key in ("exponential", "linear"):
                values = np.asarray(vectors[(noise, method, fit_key)], dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                # A series is considered pathological for display scaling when
                # any of its four circuit-family errors reaches the trigger.
                if float(np.max(values)) >= BROKEN_BAR_ERROR_THRESHOLD:
                    continue
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=AGGREGATE_STD_DDOF))
                normal_upper.append(mean + std)
                normal_lower.append(mean - std)

    if normal_upper:
        ymax = max(BROKEN_BAR_MIN_YMAX, BROKEN_BAR_HEADROOM * max(normal_upper))
        ymin_raw = min([0.0, *normal_lower]) if normal_lower else 0.0
        # Preserve a small amount of negative room only when ordinary SD bars
        # actually cross zero. Absolute-error means themselves remain >= 0.
        ymin = 1.08 * ymin_raw if ymin_raw < 0 else 0.0
    else:
        # Extremely unusual case: every series is pathological. Keep a compact
        # diagnostic window anchored to the user-selected threshold.
        ymin = 0.0
        ymax = BROKEN_BAR_ERROR_THRESHOLD
    if ymax <= ymin:
        ymax = ymin + BROKEN_BAR_MIN_YMAX
    return ymin, ymax


def _draw_bar_break(axis, *, x: float, width: float, y: float, edgecolor) -> None:
    """Draw a publication-style zigzag break across a truncated bar."""
    dx = width * 0.48
    amp = max(abs(axis.get_ylim()[1] - axis.get_ylim()[0]) * 0.012, 1e-6)
    xs = np.array([x - dx, x - dx / 2, x, x + dx / 2, x + dx])
    ys = np.array([y - amp, y + amp, y - amp, y + amp, y - amp])
    # White under-stroke creates a visible cut through the filled bar.
    axis.plot(xs, ys, color="white", linewidth=4.5, zorder=8, solid_capstyle="round")
    axis.plot(xs, ys, color=edgecolor, linewidth=1.5, zorder=9, solid_capstyle="round")


def _plot_across_circuit_bar_broken(
    vectors: dict[tuple[str, str, str], list[float]],
    *,
    methods: tuple[str, ...],
    output_path: Path,
    title_prefix: str,
    depth_summary: str,
):
    """Readable companion bar plot with pathological bars visually truncated.

    Statistics are NOT altered. Only the drawing is clipped. Truncated bars are
    marked with a zigzag and annotated with the actual mean ± SD.
    """
    figure, axis = plt.subplots(figsize=AGGREGATE_BAR_FIGSIZE)
    noises = list(NOISE_PROFILES)
    x_base = np.arange(len(noises), dtype=float)
    series = _series_order(methods)
    offsets = np.linspace(
        -AGGREGATE_CLUSTER_SPAN / 2.0,
        AGGREGATE_CLUSTER_SPAN / 2.0,
        len(series),
    )
    ymin, ymax = _broken_bar_display_limits(vectors, methods)
    yrange = ymax - ymin
    axis.set_ylim(ymin, ymax)

    legend_handles = []
    for index, (method, fit_key, fit_label, _marker) in enumerate(series):
        style = gate_plotter._method_style(method)
        facecolor = style["color"]
        edgecolor = "black" if fit_key == "linear" else _darken_color(facecolor)

        for noise_index, noise in enumerate(noises):
            values = np.asarray(vectors[(noise, method, fit_key)], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=AGGREGATE_STD_DDOF))
            pathological = float(np.max(values)) >= BROKEN_BAR_ERROR_THRESHOLD
            position = float(x_base[noise_index] + offsets[index])

            if not pathological:
                axis.bar(
                    [position], [mean],
                    width=AGGREGATE_BAR_WIDTH,
                    yerr=[std],
                    capsize=3,
                    color=facecolor,
                    edgecolor=edgecolor,
                    linewidth=1.8,
                    alpha=0.86,
                    error_kw={"elinewidth": 1.1, "capthick": 1.1},
                    zorder=3,
                )
                continue

            # Keep the real statistics, but draw the bar only inside the compact
            # display window. The numeric annotation reports the true values.
            drawn_top = ymin + 0.80 * yrange
            displayed_height = max(0.0, drawn_top - max(0.0, ymin))
            axis.bar(
                [position], [displayed_height],
                bottom=max(0.0, ymin),
                width=AGGREGATE_BAR_WIDTH,
                color=facecolor,
                edgecolor=edgecolor,
                linewidth=1.8,
                alpha=0.86,
                zorder=3,
            )
            _draw_bar_break(
                axis,
                x=position,
                width=AGGREGATE_BAR_WIDTH,
                y=drawn_top - 0.035 * yrange,
                edgecolor=edgecolor,
            )
            axis.text(
                position,
                ymin + 0.94 * yrange,
                f"{mean:.3g} ± {std:.3g}",
                ha="center",
                va="top",
                fontsize=8.5,
                rotation=90 if len(series) >= 6 else 0,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
                zorder=10,
            )

        legend_handles.append(
            Patch(
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.8,
                label=f"{style['label']} {fit_label}",
            )
        )

    axis.set_xticks(x_base)
    axis.set_xticklabels([NOISE_PROFILES[key]["label"] for key in noises])
    axis.set_ylabel(
        r"Mean absolute ZNE error $|E_{\mathrm{ZNE}}-E_{\mathrm{ideal}}|$"
        "\n(error bar: SD across 4 circuit types)"
    )
    axis.set_xlabel("Noise model")
    axis.set_title(
        f"{title_prefix}\nMean ± SD across Random, HEA-CZ, HEA-CRZ, and HEA-RZZ\n"
        f"Depths in this round: {depth_summary}\n"
        f"Truncated view: zigzag bars have individual error ≥ {BROKEN_BAR_ERROR_THRESHOLD:g}",
        fontweight="bold",
    )
    axis.grid(True, axis="y", alpha=0.3, zorder=0)
    axis.set_xlim(-0.55, len(noises) - 0.45)
    figure.tight_layout()
    _save_plot_threeways(
        figure,
        axis,
        handles=legend_handles,
        output_path=output_path,
        ncol=2,
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)


def _plot_across_circuit_box(
    vectors: dict[tuple[str, str, str], list[float]],
    *,
    methods: tuple[str, ...],
    output_path: Path,
    title_prefix: str,
    depth_summary: str,
):
    figure, axis = plt.subplots(figsize=AGGREGATE_BOX_FIGSIZE)
    noises = list(NOISE_PROFILES)
    x_base = np.arange(len(noises), dtype=float)
    series = _series_order(methods)
    offsets = np.linspace(
        -AGGREGATE_CLUSTER_SPAN / 2.0,
        AGGREGATE_CLUSTER_SPAN / 2.0,
        len(series),
    )

    legend_handles = []
    data_artists = []
    jitter = np.linspace(-0.022, 0.022, len(CIRCUIT_ORDER))
    for index, (method, fit_key, fit_label, marker) in enumerate(series):
        style = gate_plotter._method_style(method)
        facecolor = style["color"]
        edgecolor = "black" if fit_key == "linear" else _darken_color(facecolor)

        for noise_index, noise in enumerate(noises):
            values = np.asarray(vectors[(noise, method, fit_key)], dtype=float)
            position = x_base[noise_index] + offsets[index]
            box = axis.boxplot(
                [values],
                positions=[position],
                widths=AGGREGATE_BAR_WIDTH * 0.92,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                boxprops={
                    "facecolor": mcolors.to_rgba(facecolor, 0.22),
                    "edgecolor": edgecolor,
                    "linewidth": 1.7,
                },
                whiskerprops={"color": edgecolor, "linewidth": 1.2},
                capprops={"color": edgecolor, "linewidth": 1.2},
                medianprops={"color": edgecolor, "linewidth": 1.8},
            )
            for artists in box.values():
                data_artists.extend(list(artists))
            scatter_artist = axis.scatter(
                position + jitter,
                values,
                color=facecolor,
                marker=marker,
                s=style["extrapolated_marker_size"] * 0.62,
                linewidths=1.1,
                zorder=5,
            )
            data_artists.append(scatter_artist)

        legend_handles.append(
            Line2D(
                [0], [0],
                color="none",
                marker=marker,
                markerfacecolor=facecolor,
                markeredgecolor=facecolor,
                markersize=8,
                label=f"{style['label']} {fit_label}",
            )
        )

    axis.set_xticks(x_base)
    axis.set_xticklabels([NOISE_PROFILES[key]["label"] for key in noises])
    axis.set_ylabel(
        r"Absolute ZNE error $|E_{\mathrm{ZNE}}-E_{\mathrm{ideal}}|$"
        "\n(4 circuit-type values per box)"
    )
    axis.set_xlabel("Noise model")
    axis.set_title(
        f"{title_prefix}\nDistribution across Random, HEA-CZ, HEA-CRZ, and HEA-RZZ\n"
        f"Depths in this round: {depth_summary}",
        fontweight="bold",
    )
    axis.grid(True, axis="y", alpha=0.3, zorder=0)
    axis.set_xlim(-0.55, len(noises) - 0.45)
    figure.tight_layout()
    _save_plot_threeways(
        figure,
        axis,
        handles=legend_handles,
        output_path=output_path,
        ncol=2,
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)


def _generate_circuit_aggregate_plots(
    records: list[ReportRecord],
    ideal_values: dict[tuple[int, str, int], float],
    output_root: Path,
    *,
    dry_run: bool,
):
    round_lookup = _depth_round_lookup(records)
    groups, unassigned, duplicates = _build_circuit_aggregate_groups(
        records, round_lookup
    )
    generated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for group_key in sorted(groups):
        num_qubits, round_index, platform_tag = group_key

        # 1q has only the random_type circuit in this experiment.  It should
        # produce its ordinary per-case figure, but not a four-circuit mean/SD
        # or box plot.
        if num_qubits == 1:
            skipped.append({
                "group": group_key,
                "reason": "1q excluded from across-circuit statistics: only random_type is available",
            })
            continue

        circuit_records = groups[group_key]
        vectors, problems, aggregate_methods = _aggregate_error_values(circuit_records, ideal_values)
        if vectors is None:
            skipped.append({
                "group": group_key,
                "reason": "requires all 4 circuit types and at least 2 folding methods complete across all 3 target noises, with successful fits and ideal values",
                **problems,
            })
            continue

        basename = _safe_tag(
            f"{num_qubits}q_round{round_index}_{platform_tag}_acrosscircuit"
        )
        group_directory = output_root / "across_circuit_statistics" / basename
        bar_path = group_directory / f"{basename}_mean_std_bar.png"
        broken_bar_path = group_directory / f"{basename}_mean_std_bar_broken.png"
        box_path = group_directory / f"{basename}_boxplot.png"
        depth_summary = _aggregate_depth_summary(circuit_records)
        max_individual_error = _aggregate_max_individual_error(vectors)
        make_broken_bar = bool(
            np.isfinite(max_individual_error)
            and max_individual_error >= BROKEN_BAR_ERROR_THRESHOLD
        )
        title_prefix = f"{num_qubits}q | depth round {round_index} | {platform_tag}"

        stats: dict[str, Any] = {}
        for (noise, method, fit_key), values in vectors.items():
            array = np.asarray(values, dtype=float)
            stats[f"{noise}|{method}|{fit_key}"] = {
                "values_by_circuit": {
                    unitary: float(value)
                    for unitary, value in zip(CIRCUIT_ORDER, array.tolist())
                },
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=AGGREGATE_STD_DDOF)),
                "n": int(len(array)),
            }

        item = {
            "group": group_key,
            "basename": basename,
            "num_qubits": num_qubits,
            "depth_round": round_index,
            "platform_tag": platform_tag,
            "circuit_order": list(CIRCUIT_ORDER),
            "aggregate_methods": list(aggregate_methods),
            "depth_summary": depth_summary,
            "std_ddof": AGGREGATE_STD_DDOF,
            "statistics": stats,
            "max_individual_absolute_error": max_individual_error,
            "broken_bar_threshold": BROKEN_BAR_ERROR_THRESHOLD,
            "broken_bar_generated": make_broken_bar,
            "bar_output": str(bar_path),
            "bar_no_legend_output": str(bar_path.with_name(bar_path.stem + "_no_legend.png")),
            "bar_legend_output": str(bar_path.with_name(bar_path.stem + "_legend.png")),
            "broken_bar_output": str(broken_bar_path) if make_broken_bar else None,
            "broken_bar_no_legend_output": (
                str(broken_bar_path.with_name(broken_bar_path.stem + "_no_legend.png"))
                if make_broken_bar else None
            ),
            "broken_bar_legend_output": (
                str(broken_bar_path.with_name(broken_bar_path.stem + "_legend.png"))
                if make_broken_bar else None
            ),
            "box_output": str(box_path),
            "box_no_legend_output": str(box_path.with_name(box_path.stem + "_no_legend.png")),
            "box_legend_output": str(box_path.with_name(box_path.stem + "_legend.png")),
        }

        if not dry_run:
            group_directory.mkdir(parents=True, exist_ok=True)
            _plot_across_circuit_bar(
                vectors,
                methods=aggregate_methods,
                output_path=bar_path,
                title_prefix=title_prefix,
                depth_summary=depth_summary,
            )
            if make_broken_bar:
                _plot_across_circuit_bar_broken(
                    vectors,
                    methods=aggregate_methods,
                    output_path=broken_bar_path,
                    title_prefix=title_prefix,
                    depth_summary=depth_summary,
                )
            _plot_across_circuit_box(
                vectors,
                methods=aggregate_methods,
                output_path=box_path,
                title_prefix=title_prefix,
                depth_summary=depth_summary,
            )
        generated.append(item)

    return generated, skipped, unassigned, duplicates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one before-weighted method-by-noise plot per case."
    )
    parser.add_argument("--report-directory", default=REPORT_DIRECTORY)
    parser.add_argument("--output-directory", default=PLOT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_directory = Path(args.report_directory).expanduser().resolve()
    output_root = Path(args.output_directory).expanduser().resolve()
    if not report_directory.is_dir():
        raise FileNotFoundError(f"Report directory not found: {report_directory}")

    records, unreadable, ignored = _load_records(report_directory)
    if not records:
        raise ValueError(f"No matching report JSON files found in {report_directory}")

    groups, duplicates = _build_groups(records)
    ideal_values = _ideal_lookup(records)
    allow_partial = bool(ALLOW_PARTIAL_GROUPS or args.allow_partial)

    required = {
        (method, noise)
        for method in METHOD_ORDER
        for noise in NOISE_PROFILES
    }
    generated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for case_key in sorted(groups):
        num_qubits, unitary, depth, platform_tag = case_key
        available = groups[case_key]
        present = set(available)
        missing = sorted(required - present)
        methods_present = {
            method for method, _noise in present if method in METHOD_ORDER
        }
        noise_present = {
            noise for _method, noise in present if noise in NOISE_PROFILES
        }
        complete = required.issubset(present)

        # A partial figure is meaningful when at least one method has data for
        # at least two target noise models.  This also allows the 4q circuit-folding
        # cases to be plotted before the width-folding methods finish.
        method_noise_counts = {
            method: sum((method, noise) in present for noise in NOISE_PROFILES)
            for method in METHOD_ORDER
        }
        partial_is_meaningful = max(method_noise_counts.values(), default=0) >= 2
        allow_partial_for_case = allow_partial

        if (not complete and not allow_partial_for_case) or (
            not complete and allow_partial_for_case and not partial_is_meaningful
        ):
            skipped.append({
                "case": case_key,
                "available_combinations": sorted(present),
                "missing_combinations": missing,
                "reason": (
                    "incomplete group"
                    if not allow_partial_for_case
                    else "partial group has fewer than two methods or two noises"
                ),
            })
            continue

        selected_records = []
        method_keys = []
        noise_keys = []
        for noise in NOISE_PROFILES:
            for method in METHOD_ORDER:
                record = available.get((method, noise))
                if record is None:
                    continue
                selected_records.append(record)
                method_keys.append(method)
                noise_keys.append(noise)

        basename = _safe_tag(
            f"{num_qubits}q_{unitary}_{depth}_{platform_tag}_methods_acrossnoise"
        )
        group_directory = output_root / "methods_across_noise" / basename
        output_path = group_directory / f"{basename}.png"
        ideal = ideal_values.get((num_qubits, unitary, depth))

        no_legend_output_path = group_directory / f"{basename}_no_legend.png"
        legend_output_path = group_directory / f"{basename}_legend.png"
        item = {
            "comparison": "methods_across_noise",
            "basename": basename,
            "num_qubits": num_qubits,
            "unitary": unitary,
            "depth": depth,
            "platform_tag": platform_tag,
            "complete": complete,
            "ideal_expectation": ideal,
            "available_combinations": sorted(present),
            "missing_combinations": missing,
            "input_reports": [str(record.path) for record in selected_records],
            "output": str(output_path),
            "no_legend_output": str(no_legend_output_path),
            "legend_output": str(legend_output_path),
        }

        if not args.dry_run:
            group_directory.mkdir(parents=True, exist_ok=True)
            actual_path = gate_plotter.generate_method_by_noise_plot(
                [record.path for record in selected_records],
                method_keys=method_keys,
                noise_keys=noise_keys,
                noise_order=tuple(NOISE_PROFILES),
                noise_labels={
                    key: value["label"] for key, value in NOISE_PROFILES.items()
                },
                output_directory=group_directory,
                output_basename=basename,
                title_prefix=basename,
                method_order=METHOD_ORDER,
                ideal_value_override=ideal,
                save_dpi=SAVE_DPI,
                show_plots=SHOW_PLOTS,
                verbose=False,
            )
            item["output"] = str(actual_path)
        generated.append(item)

    aggregate_generated: list[dict[str, Any]] = []
    aggregate_skipped: list[dict[str, Any]] = []
    aggregate_unassigned: list[dict[str, Any]] = []
    aggregate_duplicates: list[dict[str, Any]] = []
    if GENERATE_CIRCUIT_AGGREGATE_PLOTS:
        (
            aggregate_generated,
            aggregate_skipped,
            aggregate_unassigned,
            aggregate_duplicates,
        ) = _generate_circuit_aggregate_plots(
            records,
            ideal_values,
            output_root,
            dry_run=bool(args.dry_run),
        )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dry_run": bool(args.dry_run),
        "report_directory": str(report_directory),
        "output_root": str(output_root),
        "reports_scanned": len(records),
        "plots_planned_or_generated": len(generated),
        "groups_skipped": len(skipped),
        "generated": generated,
        "skipped": skipped,
        "circuit_aggregate_generated": aggregate_generated,
        "circuit_aggregate_skipped": aggregate_skipped,
        "circuit_aggregate_unassigned": aggregate_unassigned,
        "circuit_aggregate_duplicates": aggregate_duplicates,
        "duplicates": duplicates,
        "unreadable_reports": unreadable,
        "ignored_reports": ignored,
    }
    manifest_path = output_root / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    mode = "planned" if args.dry_run else "generated"
    print(f"Reports used                 : {len(records)}")
    print(f"One-figure groups {mode:9s}: {len(generated)}")
    print(f"Incomplete groups skipped    : {len(skipped)}")
    print(f"Across-circuit groups {mode:9s}: {len(aggregate_generated)}")
    print(f"Across-circuit groups skipped: {len(aggregate_skipped)}")
    print(f"Ignored non-target reports   : {len(ignored)}")
    print(f"Manifest                     : {manifest_path}")
    if args.dry_run:
        print("\nNo PNG files were created because --dry-run was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
