import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Input reports
# ============================================================
# Put 1, 2, or 3 report filenames here. The files are read from
# INPUT_REPORT_DIRECTORY unless an absolute path is supplied.
INPUT_REPORT_DIRECTORY = "./zne_reports"
INPUT_REPORT_FILES = [
    "2q_hea_crz_36_CPU_circuit_folding.json",
    "2q_hea_crz_36_CPU_global.json",
    "2q_hea_crz_36_CPU_local.json",
]

# If one input report contains ideal_expectation, that value is used automatically.
# Otherwise set a numerical ideal value here. If both are present, they must agree.
IDEAL_VALUE_OVERRIDE = None

# False: make 5 figures (4 expectation-value figures + 1 extrapolated-summary figure).
# True: also make the 4 error figures, for 9 figures total.
# Error is defined as expectation - ideal. No additional error fitting is done;
# the existing expectation fit curves and N=0 estimates are shifted by the ideal.
PLOT_ERROR_GRAPHS = False

# ============================================================
# Plot output
# ============================================================
PLOT_OUTPUT_DIRECTORY = "./zne_plots"
SAVE_DPI = 200
SHOW_PLOTS = False

# ============================================================
# Method appearance
# ============================================================
# Change these values freely. Unknown method names use the fallback style.
METHOD_STYLES = {
    "circuit_folding": {
        "label": "Circuit folding",
        "color": "tab:blue",
        "marker": "o",
        "marker_size": 55,
        "extrapolated_marker_size": 85,
    },
    "global": {
        "label": "Global",
        "color": "tab:orange",
        "marker": "s",
        "marker_size": 55,
        "extrapolated_marker_size": 85,
    },
    "local": {
        "label": "Local",
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

LINEAR_LINESTYLE = "--"
EXPONENTIAL_LINESTYLE = "-"
LINEAR_ZERO_MARKER = "X"
EXPONENTIAL_ZERO_MARKER = "D"
IDEAL_LINE_COLOR = "black"
IDEAL_LINESTYLE = ":"
IDEAL_SUMMARY_MARKER = "*"
IDEAL_SUMMARY_MARKER_SIZE = 130
ERROR_ZERO_LINE_COLOR = "black"
ERROR_ZERO_LINESTYLE = ":"

# ============================================================
# Graph title / legend flags
# Keep these switches ABOVE the title/axis-label contents so they are easy to find.
# ============================================================
# Graph 1: expectation, before transpilation, weighted gate count
SHOW_TITLE_BEFORE_WEIGHTED = True
SHOW_LEGEND_BEFORE_WEIGHTED = True
TITLE_BEFORE_WEIGHTED = "Before transpilation: extrapolation to zero weighted gate count"
X_LABEL_BEFORE_WEIGHTED = r"$N=w_1N_{1q}+w_2N_{2q}$ before transpilation"
Y_LABEL_BEFORE_WEIGHTED = r"$\langle Z^{\otimes n}\rangle$"

# Graph 2: expectation, before transpilation, two-qubit gate count
SHOW_TITLE_BEFORE_TWO_QUBIT = True
SHOW_LEGEND_BEFORE_TWO_QUBIT = True
TITLE_BEFORE_TWO_QUBIT = "Before transpilation: extrapolation to zero two-qubit gate count"
X_LABEL_BEFORE_TWO_QUBIT = r"$N=N_{2q}$ before transpilation"
Y_LABEL_BEFORE_TWO_QUBIT = r"$\langle Z^{\otimes n}\rangle$"

# Graph 3: expectation, after transpilation, weighted gate count
SHOW_TITLE_AFTER_WEIGHTED = True
SHOW_LEGEND_AFTER_WEIGHTED = True
TITLE_AFTER_WEIGHTED = "After transpilation: extrapolation to zero weighted gate count"
X_LABEL_AFTER_WEIGHTED = r"$N=w_1N_{1q}+w_2N_{2q}$ after transpilation"
Y_LABEL_AFTER_WEIGHTED = r"$\langle Z^{\otimes n}\rangle$"

# Graph 4: expectation, after transpilation, two-qubit gate count
SHOW_TITLE_AFTER_TWO_QUBIT = True
SHOW_LEGEND_AFTER_TWO_QUBIT = True
TITLE_AFTER_TWO_QUBIT = "After transpilation: extrapolation to zero two-qubit gate count"
X_LABEL_AFTER_TWO_QUBIT = r"$N=N_{2q}$ after transpilation"
Y_LABEL_AFTER_TWO_QUBIT = r"$\langle Z^{\otimes n}\rangle$"

# Graph 5: summary of the six extrapolated values across all four N definitions.
# With three input methods, each x-axis case shows six points:
#   3 methods x {exponential, linear}.
# Each method/fit series keeps the SAME color and marker across all four cases.
SHOW_TITLE_EXTRAPOLATED_SUMMARY = True
SHOW_LEGEND_EXTRAPOLATED_SUMMARY = True
TITLE_EXTRAPOLATED_SUMMARY = "Extrapolated values for four gate-count definitions"
X_LABEL_EXTRAPOLATED_SUMMARY = "Gate-count definition"
Y_LABEL_EXTRAPOLATED_SUMMARY = r"Extrapolated $\langle Z^{\otimes n}\rangle$ at $N=0$"
SUMMARY_CASE_LABELS = {
    "before_weighted": "Before\n$N=w_1N_{1q}+w_2N_{2q}$",
    "before_two_qubit": "Before\n$N=N_{2q}$",
    "after_weighted": "After\n$N=w_1N_{1q}+w_2N_{2q}$",
    "after_two_qubit": "After\n$N=N_{2q}$",
}
# Horizontal spacing used only to separate the multiple extrapolated points
# at each categorical x position; it has no physical meaning.
SUMMARY_POINT_OFFSET_SPAN = 0.34

# ============================================================
# Optional error graphs: error = expectation - ideal
# No new fitting is performed. The already-fitted expectation curves and N=0
# estimates are simply shifted by the ideal value.
# ============================================================
SHOW_TITLE_ERROR_BEFORE_WEIGHTED = True
SHOW_ERROR_LEGEND_BEFORE_WEIGHTED = True
ERROR_TITLE_BEFORE_WEIGHTED = "Before transpilation: error vs weighted gate count"
ERROR_X_LABEL_BEFORE_WEIGHTED = X_LABEL_BEFORE_WEIGHTED
ERROR_Y_LABEL_BEFORE_WEIGHTED = "Expectation error (expectation - ideal)"

SHOW_TITLE_ERROR_BEFORE_TWO_QUBIT = True
SHOW_ERROR_LEGEND_BEFORE_TWO_QUBIT = True
ERROR_TITLE_BEFORE_TWO_QUBIT = "Before transpilation: error vs two-qubit gate count"
ERROR_X_LABEL_BEFORE_TWO_QUBIT = X_LABEL_BEFORE_TWO_QUBIT
ERROR_Y_LABEL_BEFORE_TWO_QUBIT = "Expectation error (expectation - ideal)"

SHOW_TITLE_ERROR_AFTER_WEIGHTED = True
SHOW_ERROR_LEGEND_AFTER_WEIGHTED = True
ERROR_TITLE_AFTER_WEIGHTED = "After transpilation: error vs weighted gate count"
ERROR_X_LABEL_AFTER_WEIGHTED = X_LABEL_AFTER_WEIGHTED
ERROR_Y_LABEL_AFTER_WEIGHTED = "Expectation error (expectation - ideal)"

SHOW_TITLE_ERROR_AFTER_TWO_QUBIT = True
SHOW_ERROR_LEGEND_AFTER_TWO_QUBIT = True
ERROR_TITLE_AFTER_TWO_QUBIT = "After transpilation: error vs two-qubit gate count"
ERROR_X_LABEL_AFTER_TWO_QUBIT = X_LABEL_AFTER_TWO_QUBIT
ERROR_Y_LABEL_AFTER_TWO_QUBIT = "Expectation error (expectation - ideal)"


EXPECTATION_GRAPH_CONFIGS = {
    "before_weighted": {
        "show_title": SHOW_TITLE_BEFORE_WEIGHTED,
        "title": TITLE_BEFORE_WEIGHTED,
        "xlabel": X_LABEL_BEFORE_WEIGHTED,
        "ylabel": Y_LABEL_BEFORE_WEIGHTED,
        "legend": SHOW_LEGEND_BEFORE_WEIGHTED,
    },
    "before_two_qubit": {
        "show_title": SHOW_TITLE_BEFORE_TWO_QUBIT,
        "title": TITLE_BEFORE_TWO_QUBIT,
        "xlabel": X_LABEL_BEFORE_TWO_QUBIT,
        "ylabel": Y_LABEL_BEFORE_TWO_QUBIT,
        "legend": SHOW_LEGEND_BEFORE_TWO_QUBIT,
    },
    "after_weighted": {
        "show_title": SHOW_TITLE_AFTER_WEIGHTED,
        "title": TITLE_AFTER_WEIGHTED,
        "xlabel": X_LABEL_AFTER_WEIGHTED,
        "ylabel": Y_LABEL_AFTER_WEIGHTED,
        "legend": SHOW_LEGEND_AFTER_WEIGHTED,
    },
    "after_two_qubit": {
        "show_title": SHOW_TITLE_AFTER_TWO_QUBIT,
        "title": TITLE_AFTER_TWO_QUBIT,
        "xlabel": X_LABEL_AFTER_TWO_QUBIT,
        "ylabel": Y_LABEL_AFTER_TWO_QUBIT,
        "legend": SHOW_LEGEND_AFTER_TWO_QUBIT,
    },
}

ERROR_GRAPH_CONFIGS = {
    "before_weighted": {
        "show_title": SHOW_TITLE_ERROR_BEFORE_WEIGHTED,
        "title": ERROR_TITLE_BEFORE_WEIGHTED,
        "xlabel": ERROR_X_LABEL_BEFORE_WEIGHTED,
        "ylabel": ERROR_Y_LABEL_BEFORE_WEIGHTED,
        "legend": SHOW_ERROR_LEGEND_BEFORE_WEIGHTED,
    },
    "before_two_qubit": {
        "show_title": SHOW_TITLE_ERROR_BEFORE_TWO_QUBIT,
        "title": ERROR_TITLE_BEFORE_TWO_QUBIT,
        "xlabel": ERROR_X_LABEL_BEFORE_TWO_QUBIT,
        "ylabel": ERROR_Y_LABEL_BEFORE_TWO_QUBIT,
        "legend": SHOW_ERROR_LEGEND_BEFORE_TWO_QUBIT,
    },
    "after_weighted": {
        "show_title": SHOW_TITLE_ERROR_AFTER_WEIGHTED,
        "title": ERROR_TITLE_AFTER_WEIGHTED,
        "xlabel": ERROR_X_LABEL_AFTER_WEIGHTED,
        "ylabel": ERROR_Y_LABEL_AFTER_WEIGHTED,
        "legend": SHOW_ERROR_LEGEND_AFTER_WEIGHTED,
    },
    "after_two_qubit": {
        "show_title": SHOW_TITLE_ERROR_AFTER_TWO_QUBIT,
        "title": ERROR_TITLE_AFTER_TWO_QUBIT,
        "xlabel": ERROR_X_LABEL_AFTER_TWO_QUBIT,
        "ylabel": ERROR_Y_LABEL_AFTER_TWO_QUBIT,
        "legend": SHOW_ERROR_LEGEND_AFTER_TWO_QUBIT,
    },
}

CASE_ORDER = (
    "before_weighted",
    "before_two_qubit",
    "after_weighted",
    "after_two_qubit",
)


def _load_report(filename):
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = Path(INPUT_REPORT_DIRECTORY).expanduser() / path
    if not path.is_file():
        raise FileNotFoundError(f"Report file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    return path, report


def _method_style(method):
    configured = METHOD_STYLES.get(method)
    if configured is not None:
        return dict(configured)
    style = dict(FALLBACK_METHOD_STYLE)
    style["label"] = style["label"] or method
    return style


def _safe_filename_tag(value):
    """Convert a report string into a filesystem-friendly filename tag."""
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
    return text or "unknown_backend"


def _backend_tag(report):
    """Return the execution tag used in every plot filename.

    New report examples:
      plain Aer on CPU/GPU -> CPU or GPU
      statevector          -> CPU_statevector or GPU_statevector
      fake backend         -> CPU_FakeQuebec or GPU_FakeQuebec
      real hardware        -> ibm_backend_name

    Older report schemas are still accepted through the fallback logic.
    """
    execution = report.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(
            "Report is missing the 'execution' block. Re-run the experiment with "
            "the updated main file so machine/backend information is written."
        )

    explicit_tag = execution.get("filename_tag")
    if explicit_tag:
        return _safe_filename_tag(explicit_tag)

    machine = execution.get("machine")
    simulation_device = execution.get("simulation_device")
    backend_kind = execution.get("backend_kind")
    backend_name = execution.get("backend_name")
    mode = str(execution.get("backend_mode", report.get("backend_mode", "")))

    if backend_kind == "real_hardware" or mode.startswith("real_hardware:"):
        return _safe_filename_tag(
            backend_name or mode.split(":", 1)[-1]
        )

    device = str(simulation_device or machine or "").upper()
    if device in {"CPU", "GPU"}:
        if backend_kind == "statevector" or mode == "statevector_noiseless":
            return f"{device}_statevector"
        if backend_kind == "fake_backend" or mode.startswith("fake_backend:"):
            fake_name = backend_name or mode.split(":", 1)[-1]
            return f"{device}_{_safe_filename_tag(fake_name)}"
        return device

    # Backward compatibility for reports created before CPU/GPU tags existed.
    if mode in (
        "aer_artificial_noise",
        "aer_depth_dependent_noise",
        "aer_shot_noiseless",
    ):
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


def _validate_reports(reports):
    if not reports:
        raise ValueError("INPUT_REPORT_FILES must contain at least one report.")

    reference = reports[0]
    required = {
        "num_qubits", "unitary_type", "depth", "weights",
        "expectation_values", "cases", "method", "execution",
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
        if _backend_tag(report) != _backend_tag(reference):
            raise ValueError(
                "All reports must use the same machine/backend configuration. "
                f"Got {_backend_tag(reference)!r} and {_backend_tag(report)!r}."
            )
        if report["weights"] != reference["weights"]:
            raise ValueError(
                "All reports must use the same w1/w2 values for weighted-N plots."
            )
        for case_key in CASE_ORDER:
            if case_key not in report["cases"]:
                raise ValueError(f"Report {index} is missing case {case_key!r}.")

    methods = [report["method"] for report in reports]
    if len(methods) != len(set(methods)):
        raise ValueError("Do not provide two reports for the same method in one plot run.")


def _ideal_value(reports):
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

    if IDEAL_VALUE_OVERRIDE is not None:
        override = float(IDEAL_VALUE_OVERRIDE)
        if report_ideal is not None and abs(override - report_ideal) > 1e-9:
            raise ValueError(
                "IDEAL_VALUE_OVERRIDE does not agree with ideal_expectation in the reports."
            )
        return override

    return report_ideal


def _fit_curve(fit, x_grid):
    if not fit.get("success"):
        return None
    model = fit.get("model")
    if model == "a*N + b":
        return fit["slope"] * x_grid + fit["intercept"]
    if model == "A*exp(-g*N)":
        return fit["amplitude"] * np.exp(-fit["decay_rate"] * x_grid)
    raise ValueError(f"Unknown fit model: {model}")


def _x_grid_and_max(case_key, reports):
    max_N = 0.0
    for report in reports:
        N = np.asarray(report["cases"][case_key]["N"], dtype=float)
        expectations = np.asarray(report["expectation_values"], dtype=float)
        if len(N) != len(expectations):
            raise ValueError(
                f"{report['method']} {case_key}: N and expectation lengths differ."
            )
        if len(N):
            max_N = max(max_N, float(np.max(N)))
    x_grid = np.linspace(0.0, max(max_N * 1.03, 1.0), 500)
    return x_grid, max_N


def _plot_expectation_case(case_key, reports, ideal, output_directory):
    config = EXPECTATION_GRAPH_CONFIGS[case_key]
    figure, axis = plt.subplots(figsize=(10, 6))
    x_grid, max_N = _x_grid_and_max(case_key, reports)

    for report in reports:
        method = report["method"]
        style = _method_style(method)
        label = style["label"]
        color = style["color"]
        marker = style["marker"]
        marker_size = style["marker_size"]
        extrapolated_marker_size = style["extrapolated_marker_size"]
        case = report["cases"][case_key]
        N = np.asarray(case["N"], dtype=float)
        expectations = np.asarray(report["expectation_values"], dtype=float)

        axis.scatter(
            N,
            expectations,
            color=color,
            marker=marker,
            s=marker_size,
            label=f"{label} data",
            zorder=4,
        )

        exp_fit = case["exponential"]
        exp_curve = _fit_curve(exp_fit, x_grid)
        if exp_curve is not None:
            axis.plot(
                x_grid,
                exp_curve,
                color=color,
                linestyle=EXPONENTIAL_LINESTYLE,
                label=f"{label} exp -> {exp_fit['at_N_zero']:.4f} at N=0",
            )
            axis.scatter(
                [0.0], [exp_fit["at_N_zero"]],
                color=color, marker=EXPONENTIAL_ZERO_MARKER,
                s=extrapolated_marker_size, zorder=5,
            )

        linear_fit = case["linear"]
        linear_curve = _fit_curve(linear_fit, x_grid)
        if linear_curve is not None:
            axis.plot(
                x_grid,
                linear_curve,
                color=color,
                linestyle=LINEAR_LINESTYLE,
                alpha=0.75,
                label=f"{label} linear -> {linear_fit['at_N_zero']:.4f} at N=0",
            )
            axis.scatter(
                [0.0], [linear_fit["at_N_zero"]],
                color=color, marker=LINEAR_ZERO_MARKER,
                s=extrapolated_marker_size, zorder=5,
            )

    if ideal is not None:
        axis.axhline(
            ideal,
            color=IDEAL_LINE_COLOR,
            linestyle=IDEAL_LINESTYLE,
            label=f"ideal {ideal:.4f}",
        )

    if config["show_title"]:
        axis.set_title(config["title"], fontweight="bold")
    axis.set_xlabel(config["xlabel"])
    axis.set_ylabel(config["ylabel"])
    axis.grid(True, alpha=0.3)
    axis.set_xlim(left=min(-0.02 * max(max_N, 1.0), -0.01))
    if config["legend"]:
        axis.legend()
    figure.tight_layout()

    reference = reports[0]
    backend_tag = _backend_tag(reference)
    filename = (
        f"{reference['num_qubits']}q_{reference['unitary_type']}_"
        f"{reference['depth']}_{backend_tag}_"
        f"{case_key}_expectation.png"
    )
    output_path = output_directory / filename
    figure.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)
    return output_path



def _plot_extrapolated_summary(reports, ideal, output_directory):
    """Plot the N=0 estimates for all four gate-count definitions together.

    With three reports this produces six series (3 methods x 2 fit models).
    Each series keeps one fixed marker/color across the four x-axis cases.
    The tiny horizontal offsets are visual only and prevent overlapping points.
    """
    figure, axis = plt.subplots(figsize=(11, 6))
    x_base = np.arange(len(CASE_ORDER), dtype=float)

    series_count = 2 * len(reports)
    if series_count == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -SUMMARY_POINT_OFFSET_SPAN / 2.0,
            SUMMARY_POINT_OFFSET_SPAN / 2.0,
            series_count,
        )

    series_index = 0
    for report in reports:
        method = report["method"]
        style = _method_style(method)
        label = style["label"]
        color = style["color"]
        marker_size = style["extrapolated_marker_size"]

        for fit_key, fit_label, marker in (
            ("exponential", "exp", EXPONENTIAL_ZERO_MARKER),
            ("linear", "linear", LINEAR_ZERO_MARKER),
        ):
            y_values = []
            for case_key in CASE_ORDER:
                fit = report["cases"][case_key][fit_key]
                if not fit.get("success"):
                    y_values.append(np.nan)
                else:
                    y_values.append(float(fit["at_N_zero"]))

            axis.scatter(
                x_base + offsets[series_index],
                np.asarray(y_values, dtype=float),
                color=color,
                marker=marker,
                s=marker_size,
                label=f"{label} {fit_label}",
                zorder=4,
            )
            series_index += 1

    # Summary graph: show the ideal value as the same dotted horizontal
    # reference line used in the other expectation-value graphs.
    if ideal is not None:
        axis.axhline(
            ideal,
            color=IDEAL_LINE_COLOR,
            linestyle=IDEAL_LINESTYLE,
            label=f"ideal {ideal:.4f}",
            zorder=2,
        )

    axis.set_xticks(x_base)
    axis.set_xticklabels([SUMMARY_CASE_LABELS[key] for key in CASE_ORDER])
    if SHOW_TITLE_EXTRAPOLATED_SUMMARY:
        axis.set_title(TITLE_EXTRAPOLATED_SUMMARY, fontweight="bold")
    axis.set_xlabel(X_LABEL_EXTRAPOLATED_SUMMARY)
    axis.set_ylabel(Y_LABEL_EXTRAPOLATED_SUMMARY)
    axis.grid(True, axis="y", alpha=0.3)
    axis.set_xlim(-0.5, len(CASE_ORDER) - 0.5)
    if SHOW_LEGEND_EXTRAPOLATED_SUMMARY:
        axis.legend(loc="upper right")
    figure.tight_layout()

    reference = reports[0]
    backend_tag = _backend_tag(reference)
    filename = (
        f"{reference['num_qubits']}q_{reference['unitary_type']}_"
        f"{reference['depth']}_{backend_tag}_"
        "extrapolated_summary.png"
    )
    output_path = output_directory / filename
    figure.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)
    return output_path

def _plot_error_case(case_key, reports, ideal, output_directory):
    if ideal is None:
        raise ValueError(
            "An ideal expectation value is required for error plots. Include a "
            "circuit-folding report with ideal_expectation or set IDEAL_VALUE_OVERRIDE."
        )

    config = ERROR_GRAPH_CONFIGS[case_key]
    figure, axis = plt.subplots(figsize=(10, 6))
    x_grid, max_N = _x_grid_and_max(case_key, reports)

    for report in reports:
        method = report["method"]
        style = _method_style(method)
        label = style["label"]
        color = style["color"]
        marker = style["marker"]
        marker_size = style["marker_size"]
        extrapolated_marker_size = style["extrapolated_marker_size"]
        case = report["cases"][case_key]
        N = np.asarray(case["N"], dtype=float)
        expectations = np.asarray(report["expectation_values"], dtype=float)
        errors = expectations - ideal

        axis.scatter(
            N,
            errors,
            color=color,
            marker=marker,
            s=marker_size,
            label=f"{label} error data",
            zorder=4,
        )

        # IMPORTANT: do not refit the error values. Shift the already-fitted
        # expectation curve by the ideal value.
        exp_fit = case["exponential"]
        exp_curve = _fit_curve(exp_fit, x_grid)
        if exp_curve is not None:
            exp_error_curve = exp_curve - ideal
            exp_error_at_zero = float(exp_fit["at_N_zero"] - ideal)
            axis.plot(
                x_grid,
                exp_error_curve,
                color=color,
                linestyle=EXPONENTIAL_LINESTYLE,
                label=f"{label} exp error -> {exp_error_at_zero:.4f} at N=0",
            )
            axis.scatter(
                [0.0], [exp_error_at_zero],
                color=color, marker=EXPONENTIAL_ZERO_MARKER,
                s=extrapolated_marker_size, zorder=5,
            )

        linear_fit = case["linear"]
        linear_curve = _fit_curve(linear_fit, x_grid)
        if linear_curve is not None:
            linear_error_curve = linear_curve - ideal
            linear_error_at_zero = float(linear_fit["at_N_zero"] - ideal)
            axis.plot(
                x_grid,
                linear_error_curve,
                color=color,
                linestyle=LINEAR_LINESTYLE,
                alpha=0.75,
                label=f"{label} linear error -> {linear_error_at_zero:.4f} at N=0",
            )
            axis.scatter(
                [0.0], [linear_error_at_zero],
                color=color, marker=LINEAR_ZERO_MARKER,
                s=extrapolated_marker_size, zorder=5,
            )

    axis.axhline(
        0.0,
        color=ERROR_ZERO_LINE_COLOR,
        linestyle=ERROR_ZERO_LINESTYLE,
        label="ideal error = 0",
    )

    if config["show_title"]:
        axis.set_title(config["title"], fontweight="bold")
    axis.set_xlabel(config["xlabel"])
    axis.set_ylabel(config["ylabel"])
    axis.grid(True, alpha=0.3)
    axis.set_xlim(left=min(-0.02 * max(max_N, 1.0), -0.01))
    if config["legend"]:
        axis.legend()
    figure.tight_layout()

    reference = reports[0]
    backend_tag = _backend_tag(reference)
    filename = (
        f"{reference['num_qubits']}q_{reference['unitary_type']}_"
        f"{reference['depth']}_{backend_tag}_"
        f"{case_key}_error.png"
    )
    output_path = output_directory / filename
    figure.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def main():
    loaded = [_load_report(filename) for filename in INPUT_REPORT_FILES]
    paths = [item[0] for item in loaded]
    reports = [item[1] for item in loaded]
    _validate_reports(reports)

    print("Input reports:")
    for path in paths:
        print(f"  {path}")

    ideal = _ideal_value(reports)
    if PLOT_ERROR_GRAPHS and ideal is None:
        raise ValueError(
            "PLOT_ERROR_GRAPHS=True requires an ideal expectation value. Include a "
            "report containing ideal_expectation (normally circuit folding run with "
            "artificial Aer noise) or set IDEAL_VALUE_OVERRIDE."
        )
    if ideal is None:
        print("Ideal expectation is unavailable; expectation plots will omit the ideal line.")
    else:
        print(f"Ideal expectation used for plots: {ideal:.12f}")

    output_directory = Path(PLOT_OUTPUT_DIRECTORY).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    output_paths = []

    # Four expectation-value figures. Each includes the ideal value as a dotted line.
    for case_key in CASE_ORDER:
        output_paths.append(
            _plot_expectation_case(case_key, reports, ideal, output_directory)
        )

    # Fifth expectation figure: compare the extrapolated N=0 values from all
    # four gate-count definitions in one categorical summary plot.
    output_paths.append(
        _plot_extrapolated_summary(reports, ideal, output_directory)
    )

    # Optional four error figures. No new fit is performed; expectation
    # data/curves/N=0 estimates are shifted by the same ideal value.
    if PLOT_ERROR_GRAPHS:
        for case_key in CASE_ORDER:
            output_paths.append(
                _plot_error_case(case_key, reports, ideal, output_directory)
            )

    print("\nSaved plots:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
