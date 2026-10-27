"""Batch plot generator for gate-axis ZNE JSON reports.

This script scans ``REPORT_DIRECTORY`` and automatically creates two comparison
families using the reusable plotting functions in ``plot_gate_axis_zne.py``:

1. Across methods
   circuit_folding vs global vs local for one fixed
   (num_qubits, unitary, depth, machine/backend/noise) setting.

2. Across noise models
   constant depolarizing vs increasing Pauli vs increasing Pauli + amplitude
   damping for one fixed (num_qubits, unitary, depth, machine, method) setting.
   Four ZNE-improvement figures are created (one per gate-count definition),
   each comparing raw noisy, linear-ZNE, and exponential-ZNE absolute errors.

Incomplete groups are skipped by default and written to the manifest instead of
causing the whole plotting batch to fail.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import plot_gate_axis_zne as gate_plotter


# ============================================================
# User configuration
# ============================================================
REPORT_DIRECTORY = "./zne_reports"
PLOT_OUTPUT_ROOT = "./zne_plots_batch"

GENERATE_ACROSS_METHOD = True
GENERATE_ACROSS_NOISE = True

# False means that an across-method plot requires all three methods, and an
# across-noise plot requires all three requested noise profiles.
ALLOW_PARTIAL_GROUPS = False

# Across-method groups always make 5 plots (4 expectation plots + 1 summary).
# True: add the 4 expectation-error plots to across-method groups when an ideal
# value is available. Across-noise groups always make four ZNE-improvement plots.
PLOT_ERROR_GRAPHS = False

SAVE_DPI = 200
SHOW_PLOTS = False

# Optional filters.  Leave as None to use every report found in the directory.
NUM_QUBITS_FILTER = None       # Example: [2, 3, 5]
UNITARY_FILTER = None          # Example: ["hea_cz", "hea_crz"]
DEPTH_FILTER = None            # Example: [30, 32]
EXECUTION_TAG_FILTER = None    # Example: ["GPU_constant_depolarizing"]
METHOD_FILTER = None           # Example: ["global", "local"]

METHOD_ORDER = ("circuit_folding", "global", "local")
METHOD_STYLES = {
    "circuit_folding": {
        "label": "Circuit folding",
        "color": "tab:blue",
        "marker": "o",
        "marker_size": 55,
        "extrapolated_marker_size": 85,
    },
    "global": {
        "label": "Global width folding",
        "color": "tab:orange",
        "marker": "s",
        "marker_size": 55,
        "extrapolated_marker_size": 85,
    },
    "local": {
        "label": "Local width folding",
        "color": "tab:green",
        "marker": "^",
        "marker_size": 65,
        "extrapolated_marker_size": 90,
    },
}

# The dictionary order is the plotting/legend order.
NOISE_PROFILES = {
    "constant_depolarizing": {
        "label": "Constant depolarizing",
        # Use the short tag requested in filenames such as
        # 2q_hea_crz_30_GPU_constant_acrossmethod.
        "output_tag": "constant",
        "color": "tab:blue",
        "marker": "o",
        "marker_size": 55,
        "extrapolated_marker_size": 85,
    },
    "increasing_pauli": {
        "label": "Increasing Pauli",
        "output_tag": "increasing_pauli",
        "color": "tab:green",
        "marker": "s",
        "marker_size": 55,
        "extrapolated_marker_size": 85,
    },
    "increasing_pauli_amplitude_damping": {
        "label": "Increasing Pauli + amplitude damping",
        "output_tag": "increasing_pauli_amplitude_damping",
        "color": "tab:orange",
        "marker": "^",
        "marker_size": 65,
        "extrapolated_marker_size": 90,
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

    @property
    def core_key(self) -> tuple[int, str, int]:
        return self.num_qubits, self.unitary, self.depth


def _safe_tag(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def _execution_tag_from_filename(path: Path, report: dict[str, Any]) -> str:
    """Recover the exact execution/profile tag used in the report filename."""
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


def _load_records(report_directory: Path) -> tuple[list[ReportRecord], list[dict[str, Any]]]:
    records: list[ReportRecord] = []
    unreadable: list[dict[str, Any]] = []

    for path in sorted(report_directory.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                report = json.load(handle)
            record = ReportRecord(
                path=path.resolve(),
                report=report,
                num_qubits=int(report["num_qubits"]),
                unitary=str(report["unitary_type"]),
                depth=int(report["depth"]),
                method=str(report["method"]),
                execution_tag=_execution_tag_from_filename(path, report),
            )
        except Exception as error:  # Keep one bad JSON from blocking every plot.
            unreadable.append({"path": str(path), "error": repr(error)})
            continue

        if _passes_filters(record):
            records.append(record)

    return records, unreadable


def _passes_filters(record: ReportRecord) -> bool:
    if NUM_QUBITS_FILTER is not None and record.num_qubits not in NUM_QUBITS_FILTER:
        return False
    if UNITARY_FILTER is not None and record.unitary not in UNITARY_FILTER:
        return False
    if DEPTH_FILTER is not None and record.depth not in DEPTH_FILTER:
        return False
    if EXECUTION_TAG_FILTER is not None and record.execution_tag not in EXECUTION_TAG_FILTER:
        return False
    if METHOD_FILTER is not None and record.method not in METHOD_FILTER:
        return False
    return True


def _split_noise_execution_tag(execution_tag: str) -> tuple[str, str] | None:
    # Match the longest suffix first because increasing_pauli is a prefix of
    # increasing_pauli_amplitude_damping.
    for noise_key in sorted(NOISE_PROFILES, key=len, reverse=True):
        suffix = f"_{noise_key}"
        if execution_tag.endswith(suffix):
            platform_tag = execution_tag[:-len(suffix)]
            if platform_tag:
                return platform_tag, noise_key
    return None


def _across_method_display_tag(execution_tag: str) -> str:
    split = _split_noise_execution_tag(execution_tag)
    if split is None:
        return execution_tag
    platform_tag, noise_key = split
    return f"{platform_tag}_{NOISE_PROFILES[noise_key]['output_tag']}"


def _ideal_lookup(records: list[ReportRecord]) -> dict[tuple[int, str, int], float]:
    values: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for record in records:
        ideal = record.report.get("ideal_expectation")
        if ideal is not None:
            values[record.core_key].append(float(ideal))

    result: dict[tuple[int, str, int], float] = {}
    for key, candidates in values.items():
        reference = candidates[0]
        if any(abs(value - reference) > 1e-9 for value in candidates[1:]):
            raise ValueError(
                f"Inconsistent ideal_expectation values for {key}: {candidates}"
            )
        result[key] = reference
    return result


def _method_groups(records: list[ReportRecord]):
    groups: dict[tuple[int, str, int, str], dict[str, ReportRecord]] = defaultdict(dict)
    duplicates: list[dict[str, Any]] = []
    for record in records:
        key = (*record.core_key, record.execution_tag)
        if record.method in groups[key]:
            duplicates.append({
                "comparison": "across_method",
                "key": key,
                "method": record.method,
                "paths": [str(groups[key][record.method].path), str(record.path)],
            })
        else:
            groups[key][record.method] = record
    return groups, duplicates


def _noise_groups(records: list[ReportRecord]):
    groups: dict[tuple[int, str, int, str, str], dict[str, ReportRecord]] = defaultdict(dict)
    duplicates: list[dict[str, Any]] = []
    for record in records:
        split = _split_noise_execution_tag(record.execution_tag)
        if split is None:
            continue
        platform_tag, noise_key = split
        key = (*record.core_key, platform_tag, record.method)
        if noise_key in groups[key]:
            duplicates.append({
                "comparison": "across_noise",
                "key": key,
                "noise": noise_key,
                "paths": [str(groups[key][noise_key].path), str(record.path)],
            })
        else:
            groups[key][noise_key] = record
    return groups, duplicates


def _expected_across_method_outputs(
    group_directory: Path, basename: str, make_errors: bool
) -> list[Path]:
    paths = [
        group_directory / f"{basename}_{case_key}_expectation.png"
        for case_key in gate_plotter.CASE_ORDER
    ]
    # Keep the existing across-method gate-axis sensitivity/summary figure.
    paths.append(group_directory / f"{basename}_extrapolated_summary.png")
    if make_errors:
        paths.extend(
            group_directory / f"{basename}_{case_key}_error.png"
            for case_key in gate_plotter.CASE_ORDER
        )
    return paths


def _expected_across_noise_outputs(
    group_directory: Path, basename: str
) -> list[Path]:
    return [
        group_directory / f"{basename}_{case_key}_zne_improvement.png"
        for case_key in gate_plotter.CASE_ORDER
    ]


def _run_across_method(
    groups,
    *,
    output_root: Path,
    ideal_values,
    dry_run: bool,
    allow_partial: bool,
    plot_errors: bool,
):
    manifest = []
    skipped = []
    required = set(METHOD_ORDER)

    for key in sorted(groups):
        num_qubits, unitary, depth, execution_tag = key
        available = groups[key]
        present = set(available)
        complete = required.issubset(present)
        selected_methods = [method for method in METHOD_ORDER if method in available]

        if (not complete and not allow_partial) or len(selected_methods) < 2:
            skipped.append({
                "comparison": "across_method",
                "num_qubits": num_qubits,
                "unitary": unitary,
                "depth": depth,
                "execution_tag": execution_tag,
                "available_methods": selected_methods,
                "missing_methods": sorted(required - present),
            })
            continue

        display_tag = _across_method_display_tag(execution_tag)
        basename = _safe_tag(
            f"{num_qubits}q_{unitary}_{depth}_{display_tag}_acrossmethod"
        )
        group_directory = output_root / "across_method" / basename
        ideal = ideal_values.get((num_qubits, unitary, depth))
        make_errors = bool(plot_errors and ideal is not None)
        report_paths = [available[method].path for method in selected_methods]
        styles = [dict(METHOD_STYLES[method]) for method in selected_methods]

        item = {
            "comparison": "across_method",
            "basename": basename,
            "num_qubits": num_qubits,
            "unitary": unitary,
            "depth": depth,
            "execution_tag": execution_tag,
            "methods": selected_methods,
            "complete": complete,
            "ideal_expectation": ideal,
            "input_reports": [str(path) for path in report_paths],
            "output_directory": str(group_directory),
            "expected_outputs": [
                str(path) for path in _expected_across_method_outputs(
                    group_directory, basename, make_errors
                )
            ],
        }

        if not dry_run:
            outputs = gate_plotter.generate_plots(
                report_paths,
                output_directory=group_directory,
                output_basename=basename,
                title_prefix=basename,
                series_styles=styles,
                require_same_execution=True,
                require_unique_methods=True,
                ideal_value_override=ideal,
                plot_error_graphs=make_errors,
                save_dpi=SAVE_DPI,
                show_plots=SHOW_PLOTS,
                verbose=False,
            )
            item["outputs"] = [str(path) for path in outputs]
        manifest.append(item)

    return manifest, skipped


def _run_across_noise(
    groups,
    *,
    output_root: Path,
    ideal_values,
    dry_run: bool,
    allow_partial: bool,
):
    manifest = []
    skipped = []
    noise_order = tuple(NOISE_PROFILES)
    required = set(noise_order)

    for key in sorted(groups):
        num_qubits, unitary, depth, platform_tag, method = key
        available = groups[key]
        present = set(available)
        complete = required.issubset(present)
        selected_noise = [noise for noise in noise_order if noise in available]

        if (not complete and not allow_partial) or len(selected_noise) < 2:
            skipped.append({
                "comparison": "across_noise",
                "num_qubits": num_qubits,
                "unitary": unitary,
                "depth": depth,
                "platform_tag": platform_tag,
                "method": method,
                "available_noise": selected_noise,
                "missing_noise": sorted(required - present),
            })
            continue

        basename = _safe_tag(
            f"{num_qubits}q_{unitary}_{depth}_{platform_tag}_{method}_acrossnoise"
        )
        group_directory = output_root / "across_noise" / basename
        ideal = ideal_values.get((num_qubits, unitary, depth))
        if ideal is None:
            skipped.append({
                "comparison": "across_noise",
                "num_qubits": num_qubits,
                "unitary": unitary,
                "depth": depth,
                "platform_tag": platform_tag,
                "method": method,
                "available_noise": selected_noise,
                "missing_noise": sorted(required - present),
                "reason": "ideal_expectation unavailable",
            })
            continue

        report_paths = [available[noise].path for noise in selected_noise]
        styles = [dict(NOISE_PROFILES[noise]) for noise in selected_noise]

        item = {
            "comparison": "across_noise",
            "basename": basename,
            "num_qubits": num_qubits,
            "unitary": unitary,
            "depth": depth,
            "platform_tag": platform_tag,
            "method": method,
            "noise_profiles": selected_noise,
            "complete": complete,
            "ideal_expectation": ideal,
            "input_reports": [str(path) for path in report_paths],
            "output_directory": str(group_directory),
            "plot_type": "zne_improvement_absolute_error",
            "expected_outputs": [
                str(path) for path in _expected_across_noise_outputs(
                    group_directory, basename
                )
            ],
        }

        if not dry_run:
            outputs = gate_plotter.generate_zne_improvement_plots(
                report_paths,
                output_directory=group_directory,
                output_basename=basename,
                title_prefix=basename,
                series_styles=styles,
                ideal_value_override=ideal,
                save_dpi=SAVE_DPI,
                show_plots=SHOW_PLOTS,
                verbose=False,
            )
            item["outputs"] = [str(path) for path in outputs]
        manifest.append(item)

    return manifest, skipped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically create across-method and across-noise ZNE plots."
    )
    parser.add_argument("--report-directory", default=REPORT_DIRECTORY)
    parser.add_argument("--output-directory", default=PLOT_OUTPUT_ROOT)
    parser.add_argument(
        "--mode",
        choices=("both", "across-method", "across-noise"),
        default="both",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--plot-errors",
        action="store_true",
        help="Add four expectation-error figures to across-method groups.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_directory = Path(args.report_directory).expanduser().resolve()
    output_root = Path(args.output_directory).expanduser().resolve()

    if not report_directory.is_dir():
        raise FileNotFoundError(f"Report directory not found: {report_directory}")

    records, unreadable = _load_records(report_directory)
    if not records:
        raise ValueError(f"No readable report JSON files found in {report_directory}")

    allow_partial = bool(ALLOW_PARTIAL_GROUPS or args.allow_partial)
    plot_errors = bool(PLOT_ERROR_GRAPHS or args.plot_errors)
    ideal_values = _ideal_lookup(records)

    method_groups, method_duplicates = _method_groups(records)
    noise_groups, noise_duplicates = _noise_groups(records)

    manifest: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    do_method = GENERATE_ACROSS_METHOD and args.mode in ("both", "across-method")
    do_noise = GENERATE_ACROSS_NOISE and args.mode in ("both", "across-noise")

    if do_method:
        generated, omitted = _run_across_method(
            method_groups,
            output_root=output_root,
            ideal_values=ideal_values,
            dry_run=args.dry_run,
            allow_partial=allow_partial,
            plot_errors=plot_errors,
        )
        manifest.extend(generated)
        skipped.extend(omitted)

    if do_noise:
        generated, omitted = _run_across_noise(
            noise_groups,
            output_root=output_root,
            ideal_values=ideal_values,
            dry_run=args.dry_run,
            allow_partial=allow_partial,
        )
        manifest.extend(generated)
        skipped.extend(omitted)

    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "dry_run": bool(args.dry_run),
        "report_directory": str(report_directory),
        "output_root": str(output_root),
        "reports_scanned": len(records),
        "unreadable_reports": unreadable,
        "duplicate_groups": method_duplicates + noise_duplicates,
        "plot_groups": manifest,
        "skipped_incomplete_groups": skipped,
    }
    manifest_path = output_root / (
        "plot_manifest_dry_run.json" if args.dry_run else "plot_manifest.json"
    )
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    method_count = sum(item["comparison"] == "across_method" for item in manifest)
    noise_count = sum(item["comparison"] == "across_noise" for item in manifest)
    expected_pngs = sum(len(item["expected_outputs"]) for item in manifest)

    print("=" * 88)
    print("ZNE BATCH PLOT PLAN" if args.dry_run else "ZNE BATCH PLOT RESULT")
    print(f"Reports scanned              : {len(records)}")
    print(f"Across-method plot groups    : {method_count}")
    print(f"Across-noise plot groups     : {noise_count}")
    print(f"Incomplete groups skipped    : {len(skipped)}")
    print(f"Unreadable reports           : {len(unreadable)}")
    print(f"Expected/generated PNG files : {expected_pngs}")
    print(f"Manifest                     : {manifest_path}")
    print("=" * 88)

    if skipped:
        print("\nExamples of skipped incomplete groups:")
        for item in skipped[:10]:
            print(" ", item)
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more; see the manifest.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
