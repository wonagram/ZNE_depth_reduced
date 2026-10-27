"""Plot gate-axis ZNE reports.

The manual entry point still accepts ``INPUT_REPORT_FILES`` and draws one
before-transpilation weighted-gate-count figure.  The batch driver imports the
reusable ``generate_method_by_noise_plot`` function to create one summary
figure for each (qubits, unitary, depth, machine) case.

Only the ``before_weighted`` gate-count definition is used in the current
analysis:

    N = w1 * N_1q + w2 * N_2q  before transpilation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Manual input reports
# ============================================================
INPUT_REPORT_DIRECTORY = "./zne_reports"
INPUT_REPORT_FILES = [
    "2q_hea_crz_30_GPU_constant_depolarizing_circuit_folding.json",
    "2q_hea_crz_30_GPU_constant_depolarizing_global.json",
    "2q_hea_crz_30_GPU_constant_depolarizing_local.json",
]

IDEAL_VALUE_OVERRIDE = None
PLOT_OUTPUT_DIRECTORY = "./zne_plots"
SAVE_DPI = 200
SHOW_PLOTS = False


# ============================================================
# Plot definition retained for the current analysis
# ============================================================
PRIMARY_CASE_KEY = "before_weighted"
PRIMARY_CASE_TITLE = (
    "Before transpilation: extrapolation to zero weighted gate count"
)
PRIMARY_X_LABEL = r"$N=w_1N_{1q}+w_2N_{2q}$ before transpilation"
PRIMARY_Y_LABEL = r"$\langle Z^{\otimes n}\rangle$"

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
FALLBACK_METHOD_STYLE = {
    "label": None,
    "color": "tab:gray",
    "marker": "o",
    "marker_size": 55,
    "extrapolated_marker_size": 85,
}

EXPONENTIAL_LINESTYLE = "-"
LINEAR_LINESTYLE = "--"
EXPONENTIAL_ZERO_MARKER = "D"
LINEAR_ZERO_MARKER = "X"
IDEAL_LINE_COLOR = "black"
IDEAL_LINESTYLE = ":"

# The new combined summary uses noise models on the x-axis and six point series:
# three methods x {exponential, linear}.  Points are deliberately not connected.
METHOD_NOISE_FIGSIZE = (11, 6)
METHOD_NOISE_OFFSET_SPAN = 0.38
METHOD_NOISE_X_LABEL = "Noise model"
METHOD_NOISE_Y_LABEL = (
    r"Extrapolated $\langle Z^{\otimes n}\rangle$ at $N=0$"
    "\n(before transpilation, weighted gate count)"
)

_UNSET = object()


# ============================================================
# Report helpers
# ============================================================
def _safe_filename_tag(value: Any) -> str:
    text = str(value).strip()
    for old, new in (
        (" ", "_"),
        ("(", ""),
        (")", ""),
        (":", "_"),
        ("/", "_"),
        ("\\", "_"),
    ):
        text = text.replace(old, new)
    return text or "unknown"


def _load_report(filename, input_report_directory=None):
    path = Path(filename).expanduser()
    if not path.is_absolute():
        directory = (
            INPUT_REPORT_DIRECTORY
            if input_report_directory is None
            else input_report_directory
        )
        path = Path(directory).expanduser() / path
    if not path.is_file():
        raise FileNotFoundError(f"Report file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return path.resolve(), report


def _backend_tag(report: dict[str, Any]) -> str:
    """Return a filesystem-friendly execution tag from a report."""
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("Report is missing the 'execution' block.")

    explicit_tag = execution.get("filename_tag")
    if explicit_tag:
        return _safe_filename_tag(explicit_tag)

    machine = execution.get("machine")
    simulation_device = execution.get("simulation_device")
    backend_kind = execution.get("backend_kind")
    backend_name = execution.get("backend_name")
    mode = str(execution.get("backend_mode", report.get("backend_mode", "")))

    if backend_kind == "real_hardware" or mode.startswith("real_hardware:"):
        return _safe_filename_tag(backend_name or mode.split(":", 1)[-1])

    device = str(simulation_device or machine or "").upper()
    if device in {"CPU", "GPU"}:
        if backend_kind == "statevector" or mode == "statevector_noiseless":
            return f"{device}_statevector"
        if backend_kind in {"fake_backend", "fake_statevector"} or mode.startswith(
            "fake_backend:"
        ):
            fake_name = backend_name or mode.split(":", 1)[-1]
            return f"{device}_{_safe_filename_tag(fake_name)}"
        return device

    if mode in {
        "aer_artificial_noise",
        "aer_depth_dependent_noise",
        "aer_shot_noiseless",
    }:
        return "CPU"
    if mode == "statevector_noiseless":
        return "CPU_statevector"
    if mode.startswith("fake_backend:"):
        return f"CPU_{_safe_filename_tag(backend_name or mode.split(':', 1)[1])}"
    if mode.startswith("real_hardware:"):
        return _safe_filename_tag(backend_name or mode.split(":", 1)[1])
    if backend_name is not None:
        return _safe_filename_tag(backend_name)
    return _safe_filename_tag(mode or "unknown_execution")


def _method_style(method: str) -> dict[str, Any]:
    configured = METHOD_STYLES.get(method)
    if configured is not None:
        return dict(configured)
    style = dict(FALLBACK_METHOD_STYLE)
    style["label"] = style["label"] or method
    return style


def _validate_reports(
    reports: Sequence[dict[str, Any]],
    *,
    require_same_execution: bool,
    require_unique_methods: bool,
) -> None:
    if not reports:
        raise ValueError("At least one report is required.")

    reference = reports[0]
    required = {
        "num_qubits",
        "unitary_type",
        "depth",
        "weights",
        "expectation_values",
        "cases",
        "method",
        "execution",
    }
    for index, report in enumerate(reports):
        missing = required - set(report)
        if missing:
            raise ValueError(f"Report {index} is missing keys: {sorted(missing)}")
        if report["num_qubits"] != reference["num_qubits"]:
            raise ValueError("All reports must use the same number of logical qubits.")
        if report["unitary_type"] != reference["unitary_type"]:
            raise ValueError("All reports must use the same UNITARY_TYPE.")
        if report["depth"] != reference["depth"]:
            raise ValueError("All reports must use the same U depth.")
        if require_same_execution and _backend_tag(report) != _backend_tag(reference):
            raise ValueError("All reports must use the same execution configuration.")
        if report["weights"] != reference["weights"]:
            raise ValueError("All reports must use the same weighted-N coefficients.")
        if PRIMARY_CASE_KEY not in report["cases"]:
            raise ValueError(
                f"Report {index} is missing case {PRIMARY_CASE_KEY!r}."
            )

    if require_unique_methods:
        methods = [str(report["method"]) for report in reports]
        if len(methods) != len(set(methods)):
            raise ValueError("Method-comparison reports must use unique methods.")


def _ideal_value(reports, ideal_value_override=_UNSET):
    values = [
        float(report["ideal_expectation"])
        for report in reports
        if report.get("ideal_expectation") is not None
    ]

    report_ideal = None
    if values:
        report_ideal = values[0]
        if any(abs(value - report_ideal) > 1e-9 for value in values[1:]):
            raise ValueError("Reports contain inconsistent ideal expectation values.")

    override_value = (
        IDEAL_VALUE_OVERRIDE
        if ideal_value_override is _UNSET
        else ideal_value_override
    )
    if override_value is not None:
        override = float(override_value)
        if report_ideal is not None and abs(override - report_ideal) > 1e-9:
            raise ValueError(
                "The supplied ideal-value override disagrees with the reports."
            )
        return override
    return report_ideal


def _fit_curve(fit: dict[str, Any], x_grid: np.ndarray):
    if not fit.get("success"):
        return None
    model = fit.get("model")
    if model == "a*N + b":
        return fit["slope"] * x_grid + fit["intercept"]
    if model == "A*exp(-g*N)":
        return fit["amplitude"] * np.exp(-fit["decay_rate"] * x_grid)
    raise ValueError(f"Unknown fit model: {model}")


def _compose_title(title_prefix, base_title):
    return base_title if not title_prefix else f"{title_prefix}\n{base_title}"


# ============================================================
# Manual: one before-weighted expectation-vs-N graph
# ============================================================
def _plot_before_weighted_expectation(
    reports,
    ideal,
    output_directory: Path,
    *,
    output_basename: str,
    title_prefix: str | None,
    save_dpi: int,
    show_plots: bool,
):
    figure, axis = plt.subplots(figsize=(10, 6))
    max_n = 0.0
    for report in reports:
        n_values = np.asarray(
            report["cases"][PRIMARY_CASE_KEY]["N"], dtype=float
        )
        if len(n_values):
            max_n = max(max_n, float(np.max(n_values)))
    x_grid = np.linspace(0.0, max(max_n * 1.03, 1.0), 500)

    for report in reports:
        method = str(report["method"])
        style = _method_style(method)
        case = report["cases"][PRIMARY_CASE_KEY]
        n_values = np.asarray(case["N"], dtype=float)
        expectations = np.asarray(report["expectation_values"], dtype=float)
        if len(n_values) != len(expectations):
            raise ValueError(f"{method}: N and expectation lengths differ.")

        axis.scatter(
            n_values,
            expectations,
            color=style["color"],
            marker=style["marker"],
            s=style["marker_size"],
            label=f"{style['label']} data",
            zorder=4,
        )

        exponential = case["exponential"]
        exp_curve = _fit_curve(exponential, x_grid)
        if exp_curve is not None:
            axis.plot(
                x_grid,
                exp_curve,
                color=style["color"],
                linestyle=EXPONENTIAL_LINESTYLE,
                label=(
                    f"{style['label']} exp -> "
                    f"{float(exponential['at_N_zero']):.4f} at N=0"
                ),
            )
            axis.scatter(
                [0.0],
                [float(exponential["at_N_zero"])],
                color=style["color"],
                marker=EXPONENTIAL_ZERO_MARKER,
                s=style["extrapolated_marker_size"],
                zorder=5,
            )

        linear = case["linear"]
        linear_curve = _fit_curve(linear, x_grid)
        if linear_curve is not None:
            axis.plot(
                x_grid,
                linear_curve,
                color=style["color"],
                linestyle=LINEAR_LINESTYLE,
                alpha=0.75,
                label=(
                    f"{style['label']} linear -> "
                    f"{float(linear['at_N_zero']):.4f} at N=0"
                ),
            )
            axis.scatter(
                [0.0],
                [float(linear["at_N_zero"])],
                color=style["color"],
                marker=LINEAR_ZERO_MARKER,
                s=style["extrapolated_marker_size"],
                zorder=5,
            )

    if ideal is not None:
        axis.axhline(
            ideal,
            color=IDEAL_LINE_COLOR,
            linestyle=IDEAL_LINESTYLE,
            label=f"ideal {ideal:.4f}",
        )

    axis.set_title(
        _compose_title(title_prefix, PRIMARY_CASE_TITLE), fontweight="bold"
    )
    axis.set_xlabel(PRIMARY_X_LABEL)
    axis.set_ylabel(PRIMARY_Y_LABEL)
    axis.grid(True, alpha=0.3)
    axis.set_xlim(left=min(-0.02 * max(max_n, 1.0), -0.01))
    axis.legend()
    figure.tight_layout()

    output_path = output_directory / f"{output_basename}_before_weighted_expectation.png"
    figure.savefig(output_path, dpi=save_dpi, bbox_inches="tight")
    if show_plots:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def generate_plots(
    report_files,
    *,
    input_report_directory=None,
    output_directory=None,
    output_basename=None,
    title_prefix=None,
    ideal_value_override=_UNSET,
    save_dpi=None,
    show_plots=None,
    verbose=True,
    **_ignored,
):
    """Generate only the before-weighted method-comparison graph."""
    loaded = [
        _load_report(path, input_report_directory=input_report_directory)
        for path in report_files
    ]
    paths = [item[0] for item in loaded]
    reports = [item[1] for item in loaded]
    _validate_reports(
        reports,
        require_same_execution=True,
        require_unique_methods=True,
    )
    ideal = _ideal_value(reports, ideal_value_override=ideal_value_override)

    out_dir = Path(
        PLOT_OUTPUT_DIRECTORY if output_directory is None else output_directory
    ).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = reports[0]
    basename = output_basename or _safe_filename_tag(
        f"{reference['num_qubits']}q_{reference['unitary_type']}_"
        f"{reference['depth']}_{_backend_tag(reference)}_acrossmethod"
    )

    if verbose:
        print("Input reports:")
        for path in paths:
            print(f"  {path}")

    output = _plot_before_weighted_expectation(
        reports,
        ideal,
        out_dir,
        output_basename=basename,
        title_prefix=title_prefix or basename,
        save_dpi=SAVE_DPI if save_dpi is None else int(save_dpi),
        show_plots=SHOW_PLOTS if show_plots is None else bool(show_plots),
    )
    if verbose:
        print(f"\nSaved plot:\n  {output}")
    return [output]


# ============================================================
# Batch: methods across noise in one graph
# ============================================================
def generate_method_by_noise_plot(
    report_files: Sequence[str | Path],
    *,
    method_keys: Sequence[str],
    noise_keys: Sequence[str],
    noise_order: Sequence[str],
    noise_labels: dict[str, str],
    input_report_directory=None,
    output_directory=None,
    output_basename=None,
    title_prefix=None,
    method_order: Sequence[str] = ("circuit_folding", "global", "local"),
    ideal_value_override=_UNSET,
    save_dpi=None,
    show_plots=None,
    verbose=True,
):
    """Create one before-weighted extrapolation summary.

    The x-axis contains noise models.  Each method contributes two point series:
    exponential and linear extrapolation.  Method is encoded by color and fit
    model by marker shape.  The points are not connected.
    """
    if not (len(report_files) == len(method_keys) == len(noise_keys)):
        raise ValueError(
            "report_files, method_keys, and noise_keys must have equal length."
        )

    loaded = [
        _load_report(path, input_report_directory=input_report_directory)
        for path in report_files
    ]
    paths = [item[0] for item in loaded]
    reports = [item[1] for item in loaded]
    _validate_reports(
        reports,
        require_same_execution=False,
        require_unique_methods=False,
    )

    grid: dict[tuple[str, str], dict[str, Any]] = {}
    for path, report, method, noise in zip(
        paths, reports, method_keys, noise_keys
    ):
        if str(report["method"]) != str(method):
            raise ValueError(
                f"Method mismatch for {path}: report={report['method']!r}, "
                f"supplied={method!r}."
            )
        key = (str(method), str(noise))
        if key in grid:
            raise ValueError(f"Duplicate method/noise report: {key}")
        grid[key] = report

    selected_noises = [noise for noise in noise_order if any(
        (method, noise) in grid for method in method_order
    )]
    selected_methods = [method for method in method_order if any(
        (method, noise) in grid for noise in selected_noises
    )]
    if not selected_noises or not selected_methods:
        raise ValueError("No plottable method/noise combinations were supplied.")

    ideal = _ideal_value(reports, ideal_value_override=ideal_value_override)
    out_dir = Path(
        PLOT_OUTPUT_DIRECTORY if output_directory is None else output_directory
    ).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = reports[0]
    basename = output_basename or _safe_filename_tag(
        f"{reference['num_qubits']}q_{reference['unitary_type']}_"
        f"{reference['depth']}_methods_acrossnoise"
    )

    figure, axis = plt.subplots(figsize=METHOD_NOISE_FIGSIZE)
    x_base = np.arange(len(selected_noises), dtype=float)
    series = [
        (method, fit_key, fit_label, marker)
        for method in selected_methods
        for fit_key, fit_label, marker in (
            ("exponential", "exp", EXPONENTIAL_ZERO_MARKER),
            ("linear", "linear", LINEAR_ZERO_MARKER),
        )
    ]
    offsets = (
        np.array([0.0])
        if len(series) == 1
        else np.linspace(
            -METHOD_NOISE_OFFSET_SPAN / 2.0,
            METHOD_NOISE_OFFSET_SPAN / 2.0,
            len(series),
        )
    )

    for series_index, (method, fit_key, fit_label, marker) in enumerate(series):
        style = _method_style(method)
        values = []
        for noise in selected_noises:
            report = grid.get((method, noise))
            if report is None:
                values.append(np.nan)
                continue
            fit = report["cases"][PRIMARY_CASE_KEY][fit_key]
            values.append(
                float(fit["at_N_zero"]) if fit.get("success") else np.nan
            )

        axis.scatter(
            x_base + offsets[series_index],
            np.asarray(values, dtype=float),
            color=style["color"],
            marker=marker,
            s=style["extrapolated_marker_size"],
            label=f"{style['label']} {fit_label}",
            zorder=4,
        )

    if ideal is not None:
        axis.axhline(
            ideal,
            color=IDEAL_LINE_COLOR,
            linestyle=IDEAL_LINESTYLE,
            label=f"ideal {ideal:.4f}",
            zorder=2,
        )

    axis.set_xticks(x_base)
    axis.set_xticklabels(
        [noise_labels.get(noise, noise) for noise in selected_noises]
    )
    axis.set_title(
        _compose_title(
            title_prefix or basename,
            "Method comparison across noise models using before-weighted extrapolation",
        ),
        fontweight="bold",
    )
    axis.set_xlabel(METHOD_NOISE_X_LABEL)
    axis.set_ylabel(METHOD_NOISE_Y_LABEL)
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_xlim(-0.5, len(selected_noises) - 0.5)
    axis.legend(loc="best")
    figure.tight_layout()

    output_path = out_dir / f"{basename}.png"
    figure.savefig(
        output_path,
        dpi=SAVE_DPI if save_dpi is None else int(save_dpi),
        bbox_inches="tight",
    )
    should_show = SHOW_PLOTS if show_plots is None else bool(show_plots)
    if should_show:
        plt.show()
    else:
        plt.close(figure)

    if verbose:
        print("Input reports:")
        for path in paths:
            print(f"  {path}")
        print(f"\nSaved plot:\n  {output_path}")
    return output_path


def main():
    generate_plots(
        INPUT_REPORT_FILES,
        input_report_directory=INPUT_REPORT_DIRECTORY,
        output_directory=PLOT_OUTPUT_DIRECTORY,
        ideal_value_override=IDEAL_VALUE_OVERRIDE,
        save_dpi=SAVE_DPI,
        show_plots=SHOW_PLOTS,
        verbose=True,
    )


if __name__ == "__main__":
    main()
