"""GPU2 Ver2 ZNE aggregate bar figures (increasing Pauli noise only).

Produces five figure families, each in two statistical modes:

  Figure 1: fixed n (2,3,4,5q); x-axis D1-D4 (depth rounds)
  Figure 2: fixed depth round (D1-D4); x-axis 2q-5q
  Figure 3: across all four depth rounds; x-axis 2q-5q
  Figure 4: across 2q-5q; x-axis D1-D4
  Figure 5: all fully available reports, x-axis folding method (64/64/32 reports)

Type 1: all *finite, successful* extrapolations; mean +/- population SD
        of absolute error |E_ZNE - E_ideal|. Pathological bars are visually
        truncated and zigzag-marked. Numeric mean +/- SD is shown for every
        broken bar and every cohort with out-of-range estimates.
Type 2: mean +/- population SD using only extrapolated E_ZNE in [-1, 1].
        Exclusions shown as (excluded/total), e.g. 2/16. Type 1 never
        shows an excluded/total label; it includes all finite estimates.

Each source atomic group requires all FOUR circuit families at the SAME
(n, depth round, folding method). A figure with any incomplete required bar
is skipped by default; --allow-partial-figures draws only complete bars.
No bar is ever averaged from fewer than four circuits. Figure 3 needs 16 per n/method;
Figure 4 needs 16 per depth/method, except LOCAL, which is defined only
for 2q and 3q (eight source reports per depth). Figure 5 needs EVERY
expected report for every method (64 Circuit, 64 Global, 32 Local).
Figure 5 is always skipped in both modes if any source is absent, even
if --allow-partial-figures is enabled.

Both extrapolations are retained: Linear is the solid left bar, Exponential is the hatched right bar.
Only reports named *_GPU_increasing_pauli_<method>.json are used.

Important: the supplied GPU2 report format may have ideal_expectation=None.
The script NEVER treats None as zero. Supply a JSON ideal reference mapping
(see --ideal-reference), or use reports containing ideal_expectation.

The plotting script only reads input JSON files and writes to output folder.
It never changes experiments, branches, or source reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

# ------------------------------------------------------------
# User settings
# ------------------------------------------------------------
REPORT_DIRECTORY = "./zne_reports_ver2"
PLOT_OUTPUT_ROOT = "./zne_plots_ver2_increasing_pauli"
SAVE_DPI = 200
# Text styling: same font and size on both x and y axis tick labels.
AXIS_LABEL_FONT_SIZE = 15
LEGEND_FONT_SIZE = AXIS_LABEL_FONT_SIZE
TICK_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_FAMILY = "DejaVu Sans"
TICK_LABEL_FONT_WEIGHT = "normal"
CASE_KEY = "before_weighted"
SD_DDOF = 0  # descriptive SD (population SD), as in the original batch code
BROKEN_BAR_ERROR_THRESHOLD = 4.0  # same individual-error threshold as original

METHOD_ORDER = ("circuit_folding", "global", "local")
METHOD_LABELS = {
    "circuit_folding": "Depth-folding",
    "global": "Global width-folding",
    "local": "Local width-folding",
}
METHOD_COLORS = {
    "circuit_folding": "#1982c4",
    "global": "#ff595e",
    "local": "#8ac926",
}
FIT_ORDER = ("linear", "exponential")
FIT_LABELS = {"exponential": "Exponential", "linear": "Linear"}
FIT_HATCHES = {"exponential": "///", "linear": ""}
CIRCUITS = ("random_type", "hea_cz", "hea_crz", "hea_rzz")
CIRCUIT_ABBREVIATIONS = ("Random", "CZ", "CRZ", "RZZ")
QUBITS = (2, 3, 4, 5)
ROUNDS = (1, 2, 3, 4)

# Exact depth schedule from supplied experiment_config_2.py (GPU2 Ver2).
# Do not infer rounds by ranking only the currently available report files:
# missing earlier rounds would otherwise silently shift the mapping.
DEPTHS = {
    "random_type": {2: (24,48,72,96), 3: (27,54,81,108), 4: (32,64,96,128), 5: (32,64,96,128)},
    "hea_cz":     {2: (30,60,90,120),3: (32,64,96,128),4: (32,64,96,128),5: (32,64,96,128)},
    "hea_crz":    {2: (30,60,90,120),3: (30,60,90,120),4: (30,60,90,120),5: (30,60,90,120)},
    "hea_rzz":    {2: (30,60,90,120),3: (32,64,96,128),4: (32,64,96,128),5: (32,64,96,128)},
}

REPORT_NAME = re.compile(
    r"^(?P<n>[2-5])q_(?P<unitary>random_type|hea_cz|hea_crz|hea_rzz)_"
    r"(?P<depth>\d+)_GPU_increasing_pauli_"
    r"(?P<method>circuit_folding|global|local)\.json$"
)


def allowed_methods(n: int) -> tuple[str, ...]:
    return METHOD_ORDER if n in (2, 3) else METHOD_ORDER[:2]


def expected_depth(n: int, unitary: str, round_number: int) -> int:
    return DEPTHS[unitary][n][round_number - 1]


def ideal_key(n: int, unitary: str, depth: int) -> str:
    return f"{n}q_{unitary}_{depth}"


@dataclass(frozen=True)
class Record:
    n: int
    round_number: int
    unitary: str
    depth: int
    method: str
    estimates: dict[str, float]
    ideal_in_report: float | None
    path: Path


@dataclass(frozen=True)
class Observation:
    estimate: float
    ideal: float
    error: float
    report_file: str
    unitary: str
    n: int
    round_number: int


@dataclass(frozen=True)
class BarStat:
    mean: float | None
    std: float | None
    retained: int
    excluded: int  # actually removed from the average; zero for Type 1
    out_of_range: int  # estimates outside [-1, 1], in both Type 1 and Type 2
    total: int
    max_error: float | None
    raw_errors: tuple[float, ...]


def finite_number(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def load_ideal_reference(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Ideal reference not found: {path}")
    with path.open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Ideal reference must be a JSON object: {'2q_hea_cz_30': 0.123, ...}")
    result: dict[str, float] = {}
    for key, value in obj.items():
        if value is None:  # Unfilled reference template values are acceptable.
            continue
        num = finite_number(value)
        if num is None or not -1.0 <= num <= 1.0:
            raise ValueError(f"Ideal reference {key} is not a finite expectation in [-1,1]: {value}")
        result[str(key)] = num
    return result


def load_reports(folder: Path) -> tuple[dict[tuple[int,int,str,str],Record], list[dict[str,str]], list[str]]:
    records: dict[tuple[int,int,str,str],Record] = {}
    bad: list[dict[str,str]] = []
    ignored: list[str] = []
    for path in sorted(folder.glob("*.json")):
        match = REPORT_NAME.fullmatch(path.name)
        if match is None:
            ignored.append(path.name)
            continue
        n = int(match.group("n"))
        unitary = match.group("unitary")
        depth = int(match.group("depth"))
        method = match.group("method")
        if method not in allowed_methods(n):
            ignored.append(path.name)
            continue
        try:
            if depth not in DEPTHS[unitary][n]:
                ignored.append(path.name)
                continue
            rnd = DEPTHS[unitary][n].index(depth) + 1
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if (int(data["num_qubits"]) != n or str(data["unitary_type"]) != unitary
                    or int(data["depth"]) != depth or str(data["method"]) != method):
                raise ValueError("Filename and report metadata disagree")
            if str(data.get("execution", {}).get("noise_filename_tag", "increasing_pauli")) != "increasing_pauli":
                raise ValueError("Report noise tag is not increasing_pauli")
            if "expectation_values" not in data or "fold_factors" not in data:
                raise ValueError("Missing expectation_values/fold_factors")
            if len(data["expectation_values"]) != len(data["fold_factors"]):
                raise ValueError("Incomplete report: fold factors / expectation count differ")
            case = data["cases"][CASE_KEY]
            estimates: dict[str,float] = {}
            for fit_name in FIT_ORDER:
                fit = case[fit_name]
                if not fit.get("success"):
                    raise ValueError(f"Unsuccessful {fit_name} fit")
                estimate = finite_number(fit.get("at_N_zero"))
                if estimate is None:
                    raise ValueError(f"Non-finite or absent {fit_name} at_N_zero")
                estimates[fit_name] = estimate
            reported = data.get("ideal_expectation")
            ideal = None if reported is None else finite_number(reported)
            if reported is not None and (ideal is None or not -1.0 <= ideal <= 1.0):
                raise ValueError(f"Invalid ideal expectation: {reported}")
            key = (n, rnd, unitary, method)
            if key in records:
                raise ValueError("Duplicate for this circuit, round and method")
            records[key] = Record(n, rnd, unitary, depth, method, estimates, ideal, path)
        except Exception as exc:
            bad.append({"file": str(path), "reason": str(exc)})
    return records, bad, ignored


def build_ideal_lookup(records: dict, external: dict[str,float]):
    by_case: dict[str, float] = dict(external)
    errors: list[str] = []
    for record in records.values():
        val = record.ideal_in_report
        if val is None:
            continue
        key = ideal_key(record.n, record.unitary, record.depth)
        if key in by_case and abs(by_case[key] - val) > 1e-8:
            errors.append(f"Inconsistent ideal {key}: reference={by_case[key]}, report={val}")
        by_case[key] = val
    if errors:
        raise ValueError("\n".join(errors))
    return by_case


def atomic_group(n: int, rnd: int, method: str, fit: str,
                 records: dict, ideals: dict[str,float]) -> tuple[list[Observation] | None, str | None]:
    """Require exactly four source reports, one for each circuit family."""
    if method not in allowed_methods(n):
        return None, "not defined for this n"
    values: list[Observation] = []
    for unitary in CIRCUITS:
        record = records.get((n, rnd, unitary, method))
        if record is None:
            return None, f"missing report: {n}q D{rnd} {unitary} {method}"
        ikey = ideal_key(n, unitary, record.depth)
        if ikey not in ideals:
            return None, f"missing ideal reference: {ikey}"
        ideal = ideals[ikey]
        estimate = record.estimates[fit]
        values.append(Observation(estimate, ideal, abs(estimate - ideal),
                                  record.path.name, unitary, n, rnd))
    return values, None


def aggregate(observations: list[Observation], filtered: bool) -> BarStat:
    """Mean and population SD of INDIVIDUAL absolute errors, not expectations.

    Type 1 includes every finite estimate (so zero samples are *excluded*).
    Type 2 excludes estimates outside [-1, 1].  In either type, separately
    track out_of_range so Type 1 can show a mean +/- SD label at those bars.
    """
    admissible = [o for o in observations if -1.0 <= o.estimate <= 1.0]
    selected = admissible if filtered else list(observations)
    errors = tuple(float(o.error) for o in selected)
    a = np.asarray(errors, dtype=float)
    out_of_range = len(observations) - len(admissible)
    return BarStat(
        mean=float(np.mean(a)) if len(a) else None,
        std=float(np.std(a, ddof=SD_DDOF)) if len(a) else None,
        retained=len(selected),
        excluded=out_of_range if filtered else 0,
        out_of_range=out_of_range,
        total=len(observations),
        max_error=float(np.max(a)) if len(a) else None,
        raw_errors=errors,
    )


def build_points(records: dict, ideals: dict[str,float],
                 family: int, fixed: int, mode: str):
    """Map (category, method, fit) to BarStat, requiring full cohorts per bar."""
    points: dict[tuple[int,str,str],BarStat] = {}
    skipped: list[dict[str,str]] = []
    categories = ROUNDS if family in (1, 4) else QUBITS
    for category in categories:
        for method in METHOD_ORDER:
            for fit in FIT_ORDER:
                source_groups: list[tuple[int,int]] = []
                if family == 1:
                    source_groups = [(fixed, category)]
                elif family == 2:
                    source_groups = [(category, fixed)]
                elif family == 3:
                    source_groups = [(category, rnd) for rnd in ROUNDS]
                elif family == 4:
                    # Local is only defined for 2q/3q and thus has N=8;
                    # other methods have N=16 (2q through 5q).
                    ns = (2, 3) if method == "local" else QUBITS
                    source_groups = [(n, category) for n in ns]
                observations: list[Observation] = []
                reason = None
                for n, rnd in source_groups:
                    if method not in allowed_methods(n):
                        reason = "method not defined for this n"
                        break
                    part, reason = atomic_group(n, rnd, method, fit, records, ideals)
                    if part is None:
                        break
                    observations.extend(part)
                if reason is not None:
                    skipped.append({"category": str(category), "method": method,
                                    "fit": fit, "reason": reason})
                    continue
                # Cohort cardinality is exactly 4, 8 or 16, never opportunistic.
                expected = len(source_groups) * len(CIRCUITS)
                if len(observations) != expected:
                    raise AssertionError("Unexpected aggregation cohort cardinality")
                points[(category, method, fit)] = aggregate(observations, filtered=(mode == "type2"))
    return categories, points, skipped


def required_bar_keys(family: int, fixed: int):
    """Required bars, excluding methods intentionally absent for 4q/5q."""
    cats = ROUNDS if family in (1, 4) else QUBITS
    needed = []
    for category in cats:
        if family == 1:
            methods = allowed_methods(fixed)
        elif family in (2,3):
            methods = allowed_methods(category)
        else:
            methods = METHOD_ORDER
        for method in methods:
            for fit in FIT_ORDER:
                needed.append((category,method,fit))
    return needed


def depths_note(family: int, fixed: int) -> str:
    if family == 1:
        groups = [f"D{r}=" + "/".join(str(expected_depth(fixed,u,r)) for u in CIRCUITS) for r in ROUNDS]
        return "Actual depths (Random/CZ/CRZ/RZZ): " + ";  ".join(groups)
    if family == 2:
        groups = [f"{n}q=" + "/".join(str(expected_depth(n,u,fixed)) for u in CIRCUITS) for n in QUBITS]
        return "Actual depths (Random/CZ/CRZ/RZZ): " + ";  ".join(groups)
    if family == 3:
        return "Each bar: four circuit families x four depth levels (N=16). Local only for 2q and 3q."
    if family == 4:
        return "Depth index follows circuit-specific schedules. Depth/Global: N=16 (2q-5q); Local: N=8 (2q-3q)."
    raise ValueError(family)


def figure_xlabel(family: int) -> str:
    # Deliberately no plot title: Figure type and condition are in the PNG name.
    if family in (1, 4):
        return "Depth level (D1-D4; circuit-specific depth)"
    return "Number of logical qubits"


def darker(color: str, factor: float = .57):
    return tuple(float(c * factor) for c in to_rgb(color))


def draw_wave(axis, x: float, width: float, y: float, color):
    dx = width * .48
    rng = axis.get_ylim()[1] - axis.get_ylim()[0]
    amp = max(rng * .012, 1.e-8)
    xs = np.linspace(x-dx, x+dx, 5)
    ys = np.array([y-amp,y+amp,y-amp,y+amp,y-amp])
    axis.plot(xs,ys,color="white",linewidth=4.5,zorder=9,solid_capstyle="round")
    axis.plot(xs,ys,color=color,linewidth=1.55,zorder=10,solid_capstyle="round")


def annotate_bar(axis, *, x: float, stat: BarStat, mode: str,
                 pathological: bool, ymax: float, font_size: float) -> None:
    """Use exactly ONE annotation per bar, never superimpose two labels.

    Type 1: show actual mean +/- SD when the bar is broken, OR if any
            extrapolation in its cohort is outside [-1, 1].  Never show N/M.
    Type 2: show N/M for rejected estimates, never mean +/- SD text.

    An opaque white backing keeps grid/errorbar artifacts from showing through
    characters (the earlier translucent backing caused gray ghosting).
    """
    if mode == "type1":
        if not (pathological or stat.out_of_range > 0):
            return
        assert stat.mean is not None and stat.std is not None
        label = f"{stat.mean:.3g} ± {stat.std:.3g}"
        if pathological:
            y = ymax * .965
            valign = "top"
        else:
            # Place above the actual mean +/- SD error cap when possible.
            cap = stat.mean + stat.std
            if cap < ymax * .70:
                y = cap + ymax * .035
                valign = "bottom"
            else:
                y = ymax * .965
                valign = "top"
    elif mode == "type2":
        if stat.excluded == 0:
            return
        label = f"{stat.excluded}/{stat.total}"
        if stat.retained == 0:
            y = ymax * .08
        else:
            assert stat.mean is not None and stat.std is not None
            y = min(ymax * .78, stat.mean + stat.std + ymax * .045)
        valign = "bottom"
    else:
        raise ValueError(f"Unknown plotting mode {mode!r}")

    axis.text(x, y, label, fontsize=font_size, ha="center", va=valign,
              rotation=90, color="#181818", zorder=12, clip_on=True,
              bbox={"facecolor": "white", "alpha": 1.0,
                    "edgecolor": "none", "pad": .7})


def render_figure(*, family: int, fixed: int, mode: str,
                  categories, points: dict, output: Path, save_dpi: int):
    figure, axis = plt.subplots(figsize=(14, 7.2))
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=.20, linestyle="--")
    x = np.arange(len(categories), dtype=float)
    # Keep all methods in the legend, but center each x-axis group over only
    # the folding methods defined for that logical qubit count.  In Figure 3,
    # 4q and 5q have no Local folding, so their four bars occupy the CENTER
    # rather than leaving a six-bar grid with two empty slots.
    methods_here = allowed_methods(fixed) if family == 1 else METHOD_ORDER
    series = [(method,fit) for method in methods_here for fit in FIT_ORDER]
    width = .115
    gaps = width * 1.16
    positions: dict[tuple[int,str,str], float] = {}
    for xi, category in enumerate(categories):
        method_choices = (
            allowed_methods(category) if family in (2, 3) else methods_here
        )
        category_series = [(method,fit) for method in method_choices for fit in FIT_ORDER]
        offsets = (np.arange(len(category_series)) - (len(category_series)-1)/2) * gaps
        for offset, (method,fit) in zip(offsets, category_series):
            positions[(category,method,fit)] = float(x[xi]+offset)
    finite_stats = [stat for stat in points.values() if stat.mean is not None]

    if mode == "type1":
        regular = [s for s in finite_stats if s.max_error is not None and
                   s.max_error < BROKEN_BAR_ERROR_THRESHOLD]
        if regular:
            peak = max(s.mean+s.std for s in regular)
            ymax = max(.045,peak * 1.58)
        else:
            ymax = BROKEN_BAR_ERROR_THRESHOLD
        # If some pathological points exist, keep a compact view for the rest.
        # Plotting waves conveys that the bar height is schematic, not a number.
    else:
        peak = max((s.mean+s.std for s in finite_stats), default=.045)
        ymax = max(.045,peak * 1.62)
    axis.set_ylim(0, ymax)
    ymin,ymax = axis.get_ylim()
    yspan = ymax-ymin

    for index,(method,fit) in enumerate(series):
        base_color = METHOD_COLORS[method]
        outline = darker(base_color)
        for xi,category in enumerate(categories):
            stat = points.get((category,method,fit))
            if stat is None:
                continue
            xx = positions[(category,method,fit)]
            if stat.retained == 0:
                # Do not fabricate a zero-height bar when all estimates fail.
                axis.plot([xx],[.01*yspan],marker="x",color="firebrick",markersize=5,zorder=7)
                annotate_bar(axis, x=xx, stat=stat, mode=mode,
                             pathological=False, ymax=ymax, font_size=8)
                continue
            assert stat.mean is not None and stat.std is not None
            pathological = (mode=="type1" and stat.max_error is not None and
                            stat.max_error >= BROKEN_BAR_ERROR_THRESHOLD)
            if pathological:
                display_height = ymax * .72
                axis.bar([xx],[display_height],width=width,color=base_color,
                         edgecolor=outline,linewidth=1.05,alpha=.88,
                         hatch=FIT_HATCHES[fit],zorder=3)
                draw_wave(axis,xx,width,ymax*.685,outline)
            else:
                axis.bar([xx],[stat.mean],yerr=[stat.std],width=width,
                         color=base_color,edgecolor=outline,linewidth=1.05,
                         hatch=FIT_HATCHES[fit],alpha=.88,capsize=2.6,zorder=3,
                         error_kw={"elinewidth":1.0,"capthick":1.0})
            annotate_bar(axis, x=xx, stat=stat, mode=mode,
                         pathological=pathological, ymax=ymax, font_size=8.5)

    if family in (1,4):
        labels = [f"D{c}" for c in categories]
    else:
        labels = [f"{c}q" for c in categories]
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    # Apply identical styling to every tick on BOTH axes, rather than using
    # set_xticklabels for x and tick_params for y (different code paths).
    axis.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)
    for tick_label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick_label.set_fontfamily(TICK_LABEL_FONT_FAMILY)
        tick_label.set_fontsize(TICK_LABEL_FONT_SIZE)
        tick_label.set_fontweight(TICK_LABEL_FONT_WEIGHT)
        tick_label.set_fontstyle("normal")
    # Keep scientific-notation offset text consistent if Matplotlib adds it.
    for offset_text in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
        offset_text.set_fontfamily(TICK_LABEL_FONT_FAMILY)
        offset_text.set_fontsize(TICK_LABEL_FONT_SIZE)
        offset_text.set_fontweight(TICK_LABEL_FONT_WEIGHT)
    axis.set_xlim(-.57,len(categories)-.43)
    # No plot title or top-right labels explanation; numeric excluded/total
    # annotations on the individual Type 2 bars are preserved.
    axis.set_xlabel(figure_xlabel(family), fontsize=AXIS_LABEL_FONT_SIZE, labelpad=9)
    axis.set_ylabel(r"Mean absolute ZNE error  $|E_{\mathrm{ZNE}}-E_{\mathrm{ideal}}|$",
                    fontsize=AXIS_LABEL_FONT_SIZE, labelpad=10)

    legend = [Patch(facecolor=METHOD_COLORS[method],edgecolor=darker(METHOD_COLORS[method]),
                    hatch=FIT_HATCHES[fit],label=f"{METHOD_LABELS[method]} | {FIT_LABELS[fit]}")
              for method,fit in series]
    # Figures 3 and 4 have no footnote below the x-axis.  Figures 1 and 2
    # retain their circuit-specific depth schedule annotation.
    if family in (3,4):
        figure.subplots_adjust(left=.105, right=.985, top=.965, bottom=.265)
    else:
        footnote = depths_note(family,fixed)
        figure.text(.5, .030, footnote, ha="center", va="center", fontsize=9.3, color="#4d4d4d")
        figure.subplots_adjust(left=.105, right=.985, top=.965, bottom=.32)
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.20),
                ncol=(2 if len(legend) == 4 else 3), fontsize=LEGEND_FONT_SIZE,
                columnspacing=1.2, labelspacing=.65, handlelength=1.5, frameon=False)

    # Only the standard figure with an outside legend is exported.
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output,dpi=save_dpi,bbox_inches="tight")
    plt.close(figure)



# ------------------------------------------------------------
# Figure 5: Overall folding-method summary, full reports only
# ------------------------------------------------------------
def build_figure5_statistics(records: dict, ideals: dict[str, float]):
    """Collect RAW errors once per report, never average previously averaged bars.

    Circuit/Global: 2-5q x D1-D4 x 4 circuit families = 64 each.
    Local: 2-3q x D1-D4 x 4 circuit families = 32.
    All observations and exact noiseless reference values are mandatory.
    """
    observations: dict[tuple[str, str], list[Observation]] = {}
    missing: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        ns = (2, 3) if method == "local" else QUBITS
        expected_count = len(ns) * len(ROUNDS) * len(CIRCUITS)
        for fit in FIT_ORDER:
            result: list[Observation] = []
            for n in ns:
                for rnd in ROUNDS:
                    group, reason = atomic_group(n, rnd, method, fit, records, ideals)
                    if group is None:
                        missing.append({"method": method, "fit": fit,
                                        "qubits": n, "depth_round": rnd,
                                        "reason": reason})
                    else:
                        result.extend(group)
            if len(result) != expected_count:
                # No partial cohort may contribute to Figure 5.
                if not any(d["method"] == method and d["fit"] == fit for d in missing):
                    missing.append({"method": method, "fit": fit,
                                    "reason": f"expected {expected_count} reports, got {len(result)}"})
            observations[(method, fit)] = result
    if missing:
        return {}, missing

    by_mode: dict[str, dict[tuple[str, str], BarStat]] = {}
    for mode in ("type1", "type2"):
        by_mode[mode] = {
            (method, fit): aggregate(observations[(method, fit)], filtered=(mode == "type2"))
            for method in METHOD_ORDER for fit in FIT_ORDER
        }
    return by_mode, []


def render_figure5(*, points: dict[tuple[str, str], BarStat], mode: str,
                   output: Path, save_dpi: int):
    """One folding-method cluster per x tick, Linear left, Exponential right."""
    figure, axis = plt.subplots(figsize=(13, 7.2))
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=.20, linestyle="--")
    x = np.arange(len(METHOD_ORDER), dtype=float)
    width = .255
    offsets = {"linear": -.145, "exponential": .145}
    finite_stats = [stat for stat in points.values() if stat.mean is not None]

    if mode == "type1":
        regular = [stat for stat in finite_stats if stat.max_error is not None
                   and stat.max_error < BROKEN_BAR_ERROR_THRESHOLD]
        if regular:
            ymax = max(.045, max(stat.mean + stat.std for stat in regular) * 1.68)
        else:
            ymax = BROKEN_BAR_ERROR_THRESHOLD
    else:
        ymax = max(.045, max((stat.mean + stat.std for stat in finite_stats),
                              default=.045) * 1.70)
    axis.set_ylim(0, ymax)

    for idx, method in enumerate(METHOD_ORDER):
        color = METHOD_COLORS[method]
        edgecolor = darker(color)
        for fit in FIT_ORDER:
            stat = points[(method, fit)]
            xx = float(x[idx] + offsets[fit])
            if stat.retained == 0:
                # All estimators excluded: no fabricated zero-error bar.
                axis.plot([xx], [.016*ymax], marker="x", markersize=7,
                          color="firebrick", zorder=7)
                annotate_bar(axis, x=xx, stat=stat, mode=mode,
                             pathological=False, ymax=ymax, font_size=11.5)
                continue
            assert stat.mean is not None and stat.std is not None
            pathological = (mode == "type1" and stat.max_error is not None
                            and stat.max_error >= BROKEN_BAR_ERROR_THRESHOLD)
            if pathological:
                axis.bar([xx], [ymax*.72], width=width, color=color,
                         edgecolor=edgecolor, linewidth=1.2, alpha=.88,
                         hatch=FIT_HATCHES[fit], zorder=3)
                draw_wave(axis, xx, width, ymax*.685, edgecolor)
            else:
                axis.bar([xx], [stat.mean], yerr=[stat.std], width=width,
                         color=color, edgecolor=edgecolor, linewidth=1.2,
                         hatch=FIT_HATCHES[fit], alpha=.88, capsize=4, zorder=3,
                         error_kw={"elinewidth": 1.15, "capthick": 1.1})
            annotate_bar(axis, x=xx, stat=stat, mode=mode,
                         pathological=pathological, ymax=ymax, font_size=11.5)

    axis.set_xticks(x)
    axis.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER])
    axis.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONT_SIZE)
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_fontfamily(TICK_LABEL_FONT_FAMILY)
        label.set_fontsize(TICK_LABEL_FONT_SIZE)
        label.set_fontweight(TICK_LABEL_FONT_WEIGHT)
        label.set_fontstyle("normal")
    for label in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
        label.set_fontfamily(TICK_LABEL_FONT_FAMILY)
        label.set_fontsize(TICK_LABEL_FONT_SIZE)
        label.set_fontweight(TICK_LABEL_FONT_WEIGHT)
    axis.set_xlim(-.58, len(METHOD_ORDER)-.42)
    axis.set_xlabel("Folding method", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=9)
    axis.set_ylabel(r"Mean absolute ZNE error  $|E_{\mathrm{ZNE}}-E_{\mathrm{ideal}}|$",
                    fontsize=AXIS_LABEL_FONT_SIZE, labelpad=10)

    legend = [Patch(facecolor="#9a9a9a", edgecolor="#474747",
                    hatch=FIT_HATCHES[fit], label=FIT_LABELS[fit])
              for fit in FIT_ORDER]
    figure.subplots_adjust(left=.12, right=.985, top=.965, bottom=.23)
    axis.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, -.19),
                ncol=2, fontsize=LEGEND_FONT_SIZE, columnspacing=2.0,
                handlelength=1.7, frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=save_dpi, bbox_inches="tight")
    plt.close(figure)


def default_ideal_template() -> dict[str,None]:
    return {ideal_key(n,u,expected_depth(n,u,r)):None
            for n in QUBITS for u in CIRCUITS for r in ROUNDS}


def write_depth_schedule(output_root: Path):
    with (output_root/"depth_schedule.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["logical_qubits","depth_level","random_type","hea_cz","hea_crz","hea_rzz"])
        for n in QUBITS:
            for r in ROUNDS:
                w.writerow([n,r,*[expected_depth(n,u,r) for u in CIRCUITS]])


def run(args) -> int:
    source = Path(args.report_directory).expanduser().resolve()
    target = Path(args.output_directory).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Report directory not found: {source}")
    if source == target:
        raise ValueError("Input and output directories must differ")
    target.mkdir(parents=True,exist_ok=True)
    records,bad,ignored=load_reports(source)
    external=load_ideal_reference(Path(args.ideal_reference).expanduser().resolve()
                                  if args.ideal_reference else None)
    ideals=build_ideal_lookup(records,external)
    template=default_ideal_template()
    not_available={key:None for key in template if key not in ideals}
    if not_available:
        template_path=target/"missing_ideal_reference_template.json"
        template_path.write_text(json.dumps(not_available,indent=2)+"\n",encoding="utf-8")
    write_depth_schedule(target)

    figures=[]
    skipped_bars=[]
    rows=[]
    for family in (1,2,3,4):
        fixed_options=QUBITS if family==1 else ROUNDS if family==2 else (0,)
        for fixed in fixed_options:
            for mode in ("type1","type2"):
                categories,points,skipped=build_points(records,ideals,family,fixed,mode)
                name=(f"figure{family}_{fixed}q" if family==1 else
                      f"figure{family}_D{fixed}" if family==2 else f"figure{family}")
                basename=f"{name}_{mode}_increasing_pauli"
                destination=target/f"figure_{family}"/(basename+".png")
                for item in skipped:
                    skipped_bars.append({"figure":name,"mode":mode,**item})
                missing_required = [key for key in required_bar_keys(family,fixed) if key not in points]
                if not points or (missing_required and not args.allow_partial_figures):
                    figures.append({"figure":name,"mode":mode,
                                    "status":"skipped_incomplete_or_missing_ideal",
                                    "n_bars":0,"missing_required_bars":
                                    [list(k) for k in missing_required]})
                    continue
                render_figure(family=family,fixed=fixed,mode=mode,
                              categories=categories,points=points,
                              output=destination,save_dpi=args.dpi)
                figures.append({"figure":name,"mode":mode,"status":"generated",
                                "n_bars":len(points),"path":str(destination)})
                for (category,method,fit),stat in sorted(points.items()):
                    rows.append({"figure":name,"mode":mode,"x_category":category,
                                 "folding_method":method,"fit":fit,
                                 "n_total":stat.total,"n_excluded":stat.excluded,
                                 "n_retained":stat.retained,
                                 "mean_abs_error":stat.mean,"std_abs_error":stat.std,
                                 "max_retained_error":stat.max_error})
    # Figure 5 uses raw errors from every required source, not means from
    # Figures 1-4.  Both types are skipped together unless all 160 expected
    # source reports and their noiseless ideal references are present.
    figure5_by_mode, figure5_missing = build_figure5_statistics(records, ideals)
    if figure5_missing:
        for mode in ("type1", "type2"):
            figures.append({"figure": "figure5", "mode": mode,
                            "status": "skipped_incomplete_or_missing_ideal",
                            "n_bars": 0, "missing_required_bars": figure5_missing})
        # Remove only previously generated Figure 5 PNGs from this output
        # directory, so rerunning against an incomplete dataset cannot leave
        # a misleading stale "complete" summary behind.
        for mode in ("type1", "type2"):
            old_figure5_png = target / "figure_5" / f"figure5_{mode}_increasing_pauli.png"
            if old_figure5_png.is_file():
                old_figure5_png.unlink()
    else:
        for mode in ("type1", "type2"):
            points5 = figure5_by_mode[mode]
            destination = target / "figure_5" / f"figure5_{mode}_increasing_pauli.png"
            render_figure5(points=points5, mode=mode, output=destination,
                           save_dpi=args.dpi)
            figures.append({"figure": "figure5", "mode": mode,
                            "status": "generated", "n_bars": len(points5),
                            "path": str(destination)})
            for (method, fit), stat in points5.items():
                rows.append({"figure": "figure5", "mode": mode,
                             "x_category": method, "folding_method": method,
                             "fit": fit, "n_total": stat.total,
                             "n_excluded": stat.excluded, "n_retained": stat.retained,
                             "mean_abs_error": stat.mean, "std_abs_error": stat.std,
                             "max_retained_error": stat.max_error})

    with (target/"aggregate_statistics.csv").open("w",newline="",encoding="utf-8") as f:
        fieldnames=["figure","mode","x_category","folding_method","fit","n_total",
                    "n_excluded","n_retained","mean_abs_error","std_abs_error","max_retained_error"]
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest={"report_directory":str(source),"output_directory":str(target),
              "noise":"increasing_pauli only", "case_key":CASE_KEY,
              "statistic":"mean absolute extrapolation error +/- population SD",
              "out_of_range_rule":"exclude only extrapolated expectation outside [-1,1] for type2",
              "local_method_cohort_note":"Local 2q/3q only; Figure 4 Local N=8 vs other methods N=16",
              "reports_used":len(records),"bad_reports":bad,"ignored_report_count":len(ignored),
              "figure5_expected_source_reports": {"circuit_folding":64,"global":64,"local":32},
              "figure5_requires_complete_cohorts": True,
              "ideal_values_available":len(ideals),
              "missing_ideal_reference_count":len(not_available),
              "figures":figures,"skipped_bars":skipped_bars}
    (target/"plot_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    made=sum(x["status"]=="generated" for x in figures)
    print(f"Reports used: {len(records)} | malformed/unsuccessful: {len(bad)} | other files ignored: {len(ignored)}")
    print(f"Ideal references available: {len(ideals)} | missing: {len(not_available)}")
    print(f"Figures: {made} / 22 (one PNG per figure, with outside legend)")
    if figure5_missing:
        print(f"Figure 5 skipped (Type1 and Type2): {len(figure5_missing)} required source group(s) missing.")
    print(f"Statistics CSV: {target/'aggregate_statistics.csv'}")
    print(f"Manifest: {target/'plot_manifest.json'}")
    if not_available:
        print(f"MISSING IDEALS: Fill {target/'missing_ideal_reference_template.json'} with actual noiseless expectations")
        print("Then run again with --ideal-reference <filled_JSON> . None is never taken as zero.")
    if bad:
        print(f"WARNING: {len(bad)} invalid/unsuccessful report(s); see plot_manifest.json")
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report-directory",default=REPORT_DIRECTORY)
    parser.add_argument("--output-directory",default=PLOT_OUTPUT_ROOT)
    parser.add_argument("--ideal-reference",default=None,
                        help="JSON object keyed by n-qubits/unitary/depth, e.g. {'2q_hea_cz_30':0.12}")
    parser.add_argument("--allow-partial-figures",action="store_true",
                        help="Default: skip an entire figure if any required 4-circuit group is missing. "
                             "This option draws only the complete bars as a diagnostic.")
    parser.add_argument("--dpi",type=int,default=SAVE_DPI)
    args=parser.parse_args()
    return run(args)


if __name__=="__main__":
    sys.exit(main())
