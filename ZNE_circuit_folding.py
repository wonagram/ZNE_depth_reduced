from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import matplotlib.pyplot as plt

import json
from pathlib import Path
from scipy.optimize import curve_fit
from qiskit.circuit import Gate

from U_config import (
    get_unitary_max_depth,
    normalize_unitary_type,
    select_unitary_components,
    selected_unitary_summary,
)
from unitary_tools import (
    NoiseController,
    append_unitary_block,
    configure_aer_simulator_device,
    make_fake_aer_backend,
)
from state_preparation import *

from zne_cli import (
    override,
    parse_cli_args,
    resolve_backend_flags,
    resolve_machine_flags,
    resolve_noise_flags,
)

_CLI_ARGS = parse_cli_args('standard circuit-folding ZNE experiment')

# ============================================================
# Execution machine
# ============================================================
# Real hardware and local simulation are mutually exclusive.
# For Aer simulation, USE_GPU=True selects Aer device="GPU".
# When USE_GPU=False, local simulation uses the CPU.
USE_REAL_HARDWARE, USE_GPU = resolve_machine_flags(
    _CLI_ARGS.machine,
    default_use_real_hardware=False,
    default_use_gpu=False,
)

REAL_HARDWARE_DEVICE_NAME = override(
    False,
    _CLI_ARGS.real_hardware_device_name,
)
# REAL_HARDWARE_DEVICE_NAME = "ibm_strasbourg"

# ============================================================
# Simulation backend
# ============================================================
# These flags apply only when USE_REAL_HARDWARE=False.
# Select exactly one of the three simulation backends.
(
    USE_AER_SIMULATOR,
    USE_STATEVECTOR,
    USE_FAKE_BACKEND,
    USE_FAKE_STATEVECTOR,
) = resolve_backend_flags(
    _CLI_ARGS.backend,
    use_real_hardware=USE_REAL_HARDWARE,
    default_use_aer_simulator=True,
    default_use_statevector=False,
    default_use_fake_backend=False,
    default_use_fake_statevector=False,
)

# Fake-backend mode uses an IBM fake-device snapshot to configure Aer.
# The snapshot supplies the device noise model; custom channels are disabled.
FAKE_BACKEND_NAME = override(
    "FakeQuebec",
    _CLI_ARGS.fake_backend_name,
)
# Available modern examples (qiskit_ibm_runtime.fake_provider):
#   FakeSherbrooke, FakeBrisbane, FakeKyiv, FakeKyoto, FakeOsaka,
#   FakeQuebec, FakeStrasbourg, FakeTorino, FakeBrussels, FakeKawasaki,
#   FakeCusco, FakeFez, FakeMarrakesh, FakeAachen, FakeBerlin,
#   FakeBoston, FakeKingston, FakeMiami, FakePittsburgh
#   FakeFractionalBackend explicitly models control flow, but has only 5 qubits.

_SIMULATION_BACKEND_FLAGS = {
    "USE_AER_SIMULATOR": USE_AER_SIMULATOR,
    "USE_STATEVECTOR": USE_STATEVECTOR,
    "USE_FAKE_BACKEND": USE_FAKE_BACKEND,
}
_ENABLED_SIMULATION_BACKENDS = [
    name for name, enabled in _SIMULATION_BACKEND_FLAGS.items() if enabled
]

if USE_REAL_HARDWARE:
    if USE_GPU:
        raise ValueError(
            "USE_GPU cannot be combined with USE_REAL_HARDWARE. GPU/CPU selects "
            "the machine for Aer simulation, not the IBM quantum device."
        )
    if _ENABLED_SIMULATION_BACKENDS:
        raise ValueError(
            "When USE_REAL_HARDWARE=True, turn off USE_AER_SIMULATOR, "
            "USE_STATEVECTOR, and USE_FAKE_BACKEND. Enabled simulation backends: "
            + ", ".join(_ENABLED_SIMULATION_BACKENDS)
        )
    SIMULATION_DEVICE = None
else:
    if len(_ENABLED_SIMULATION_BACKENDS) != 1:
        raise ValueError(
            "When USE_REAL_HARDWARE=False, select exactly one simulation backend: "
            "USE_AER_SIMULATOR, USE_STATEVECTOR, or USE_FAKE_BACKEND. Enabled: "
            + (", ".join(_ENABLED_SIMULATION_BACKENDS)
               if _ENABLED_SIMULATION_BACKENDS else "none")
        )
    SIMULATION_DEVICE = "GPU" if USE_GPU else "CPU"

shots = int(override(
    10**3,
    _CLI_ARGS.shots,
))  # Used by shot-based Aer/fake/hardware; ignored by statevector.

# ============================================================
# Gate-count-axis extrapolation report
# ============================================================
# Four extrapolation axes are always reported:
#   1) before transpilation: N = w1*N_1q + w2*N_2q
#   2) before transpilation: N = N_2q
#   3) after transpilation:  N = w1*N_1q + w2*N_2q
#   4) after transpilation:  N = N_2q
#
# The report is JSON and is overwritten when the same qubit/unitary/method
# combination is run again.
REPORT_DIRECTORY = str(override(
    "./zne_reports",
    _CLI_ARGS.report_directory,
))
GATE_WEIGHT_1Q = float(override(0.3, _CLI_ARGS.gate_weight_1q))
GATE_WEIGHT_2Q = float(override(0.7, _CLI_ARGS.gate_weight_2q))

# ============================================================
# Unitary selection, counted depth, and logical-qubit count
# Available UNITARY_TYPE values:
#   random_type, hea_cz, hea_crx, hea_crz, hea_rzz,
#   qaoa, qaoa_xy, clifford_only
# ============================================================
UNITARY_TYPE = str(override("random_type", _CLI_ARGS.unitary))
DEPTH = int(override(18, _CLI_ARGS.depth))
NUM_U_QUBITS = int(override(2, _CLI_ARGS.num_qubits))

# ============================================================
# Circuit drawing
# ============================================================
DRAW_CIRCUIT = False
DRAW_DEPTH_FOLDED = 5

# ============================================================
# Number of foldings
# ============================================================
depth_folded_circuits = list(override(
    [1, 3, 5, 7],
    _CLI_ARGS.fold_factors,
))

# ============================================================
# Custom noise configuration for USE_AER_SIMULATOR
#
# Historical constant-noise mode (preserved):
#   select any combination of the four USE_CONSTANT_* flags.
#
# New depth-dependent mode:
#   select USE_INCREASING_RANDOM_PAULI and/or
#   USE_INCREASING_AMPLITUDE_DAMPING.
#
# The historical and depth-dependent families are mutually exclusive.
# Statevector, fake-backend, and real-hardware modes cannot be combined with
# custom-noise flags.  Aer with every custom-noise flag False is noiseless,
# shot-based Aer and remains available for backward compatibility.
# ============================================================
(
    USE_CONSTANT_DEPOLARIZING,
    USE_CONSTANT_AMPLITUDE_DAMPING,
    USE_CONSTANT_PHASE_DAMPING,
    USE_CONSTANT_COHERENT_OVERROTATION,
    USE_INCREASING_RANDOM_PAULI,
    USE_INCREASING_AMPLITUDE_DAMPING,
) = resolve_noise_flags(
    _CLI_ARGS.noise,
    defaults=(True, False, False, False, False, False),
)

# Historical constant-noise strengths (same semantics as before).
depolarizing_strength = float(override(0.02, _CLI_ARGS.depolarizing_strength))
amplitude_damping_strength = 0.02
phase_damping_strength = 0.02
overrotation_epsilon = 0.02

# Reproducible depth-dependent random-Pauli assignments and Aer sampling.
BASE_SEED = int(override(150, _CLI_ARGS.base_seed))

# Capped depth-dependent noise model.
#
# For every original-gate noise location at structural depth D, use
#
#   scale(D) = min(
#       1 + (NOISE_MULTIPLIER - 1) * D / REFERENCE_DEPTH,
#       MAX_NOISE_MULTIPLIER,
#   )
#   parameter(D) = parameter_initial * scale(D).
#
# With the values below, scale(D) = min(1 + 2D/100, 4):
#   D =   0 -> 1x,
#   D = 100 -> 3x,
#   D >=150 -> 4x cap.
#
# At the 4x cap, the four error parameters stop at:
#   p_1q     = 0.000980000 = 0.0980000%
#   p_2q     = 0.027880000 = 2.7880000%
#   gamma_1q = 0.000448692 = 0.0448692%
#   gamma_2q = 0.008452596 = 0.8452596%
#
# The cap is intentional: folded circuits may reach depths far beyond the
# original U depth, but their per-location error probabilities must not keep
# increasing without bound.
REFERENCE_DEPTH = 100
NOISE_MULTIPLIER = 3.0
MAX_NOISE_MULTIPLIER = 4.0
NOISE_CAP_DEPTH = (
    REFERENCE_DEPTH
    * (MAX_NOISE_MULTIPLIER - 1.0)
    / (NOISE_MULTIPLIER - 1.0)
)

P1_INITIAL = 0.000245
P2_INITIAL = 0.006970
GAMMA1_INITIAL = 0.000112173
GAMMA2_INITIAL = 0.002113149

# Linear increments before the cap is reached.
P1_INCREMENT_PER_DEPTH = (
    (NOISE_MULTIPLIER - 1.0) * P1_INITIAL / REFERENCE_DEPTH
)
P2_INCREMENT_PER_DEPTH = (
    (NOISE_MULTIPLIER - 1.0) * P2_INITIAL / REFERENCE_DEPTH
)
GAMMA1_INCREMENT_PER_DEPTH = (
    (NOISE_MULTIPLIER - 1.0) * GAMMA1_INITIAL / REFERENCE_DEPTH
)
GAMMA2_INCREMENT_PER_DEPTH = (
    (NOISE_MULTIPLIER - 1.0) * GAMMA2_INITIAL / REFERENCE_DEPTH
)

# Explicit maximum values recorded here for readability and validation.
P1_MAX = P1_INITIAL * MAX_NOISE_MULTIPLIER
P2_MAX = P2_INITIAL * MAX_NOISE_MULTIPLIER
GAMMA1_MAX = GAMMA1_INITIAL * MAX_NOISE_MULTIPLIER
GAMMA2_MAX = GAMMA2_INITIAL * MAX_NOISE_MULTIPLIER

_CONSTANT_NOISE_ACTIVE = any((
    USE_CONSTANT_DEPOLARIZING,
    USE_CONSTANT_AMPLITUDE_DAMPING,
    USE_CONSTANT_PHASE_DAMPING,
    USE_CONSTANT_COHERENT_OVERROTATION,
))
_INCREASING_NOISE_ACTIVE = any((
    USE_INCREASING_RANDOM_PAULI,
    USE_INCREASING_AMPLITUDE_DAMPING,
))
_CUSTOM_NOISE_ACTIVE = _CONSTANT_NOISE_ACTIVE or _INCREASING_NOISE_ACTIVE

if _CONSTANT_NOISE_ACTIVE and _INCREASING_NOISE_ACTIVE:
    raise ValueError(
        "Do not enable historical constant noise and depth-dependent noise "
        "simultaneously. Choose one custom-noise family."
    )

if _CUSTOM_NOISE_ACTIVE and not USE_AER_SIMULATOR:
    raise ValueError(
        "Custom constant/depth-dependent noise is available only with "
        "USE_AER_SIMULATOR=True. Turn off custom-noise flags when using "
        "statevector, a fake backend, or real hardware."
    )

# ============================================================
# Select U. DEPTH may cut any decomposed macro at a physical-depth boundary.
# ============================================================
UNITARY_TYPE = normalize_unitary_type(UNITARY_TYPE)
SELECTED_COMPONENTS = select_unitary_components(
    UNITARY_TYPE,
    NUM_U_QUBITS,
    DEPTH,
)
MAX_U_DEPTH = get_unitary_max_depth(UNITARY_TYPE, NUM_U_QUBITS)

NOISE_CONTROLLER = NoiseController(
    # Historical constant mode.
    use_depolarizing=USE_CONSTANT_DEPOLARIZING,
    use_amplitude_damping=USE_CONSTANT_AMPLITUDE_DAMPING,
    use_phase_damping=USE_CONSTANT_PHASE_DAMPING,
    use_coherent_overrotation=USE_CONSTANT_COHERENT_OVERROTATION,
    depolarizing_strength=depolarizing_strength,
    amplitude_damping_strength=amplitude_damping_strength,
    phase_damping_strength=phase_damping_strength,
    overrotation_epsilon=overrotation_epsilon,
    # New depth-dependent mode.
    use_increasing_random_pauli=USE_INCREASING_RANDOM_PAULI,
    use_increasing_amplitude_damping=USE_INCREASING_AMPLITUDE_DAMPING,
    base_seed=BASE_SEED,
    reference_depth=REFERENCE_DEPTH,
    noise_multiplier=NOISE_MULTIPLIER,
    maximum_noise_multiplier=MAX_NOISE_MULTIPLIER,
    p1_initial=P1_INITIAL,
    p2_initial=P2_INITIAL,
    gamma1_initial=GAMMA1_INITIAL,
    gamma2_initial=GAMMA2_INITIAL,
)


# ============================================================
# Gate counting, fitting, and JSON reporting
# ============================================================
_SKIP_COUNT_NAMES = {
    "barrier", "delay", "measure", "reset",
    "kraus", "superop", "quantum_channel", "pauli",
    "save_expectation_value", "save_statevector", "set_statevector",
}


def _count_one_two_qubit_gates(circuit, _level=0):
    """Count structural one- and two-qubit gates in a circuit.

    Control-flow blocks are inspected recursively.  Noise-channel instructions,
    measurement/reset, barriers, delays, and Aer save/set instructions are not
    counted as gates.  A composite operation acting on more than two qubits is
    recursively expanded when it exposes a definition.
    """
    if _level > 30:
        raise RuntimeError("Gate-count recursion exceeded 30 levels.")

    n1q = 0
    n2q = 0

    for item in circuit.data:
        op = item.operation
        name = getattr(op, "name", "")

        if getattr(op, "label", None) == "__zne_artificial_noise__":
            continue

        blocks = tuple(getattr(op, "blocks", ()) or ())
        if blocks:
            for block in blocks:
                c1, c2 = _count_one_two_qubit_gates(block, _level + 1)
                n1q += c1
                n2q += c2
            continue

        if name in _SKIP_COUNT_NAMES:
            continue

        if isinstance(op, Gate):
            width = len(item.qubits)
            if width == 1:
                n1q += 1
            elif width == 2:
                n2q += 1
            else:
                definition = getattr(op, "definition", None)
                if definition is not None:
                    c1, c2 = _count_one_two_qubit_gates(definition, _level + 1)
                    n1q += c1
                    n2q += c2
            continue

        definition = getattr(op, "definition", None)
        if definition is not None:
            c1, c2 = _count_one_two_qubit_gates(definition, _level + 1)
            n1q += c1
            n2q += c2

    return int(n1q), int(n2q)


def _gate_count_list(circuits):
    return [
        {"N_1q": n1q, "N_2q": n2q}
        for n1q, n2q in (_count_one_two_qubit_gates(circuit) for circuit in circuits)
    ]


def _linear_fit(N_values, expectations):
    x = np.asarray(N_values, dtype=float)
    y = np.asarray(expectations, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return {
            "success": False,
            "error": "Linear fitting requires at least two distinct N values.",
        }

    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
    return {
        "success": True,
        "model": "a*N + b",
        "slope": float(slope),
        "intercept": float(intercept),
        "at_N_zero": float(intercept),
        "rmse": rmse,
    }


def _exponential_model(N, amplitude, decay_rate):
    return amplitude * np.exp(-decay_rate * N)


def _exponential_fit(N_values, expectations):
    x = np.asarray(N_values, dtype=float)
    y = np.asarray(expectations, dtype=float)
    if len(x) < 2 or len(np.unique(x)) < 2:
        return {
            "success": False,
            "error": "Exponential fitting requires at least two distinct N values.",
        }

    smallest_index = int(np.argmin(x))
    amplitude0 = float(y[smallest_index])
    if abs(amplitude0) < 1e-12:
        amplitude0 = float(np.mean(y))
    if abs(amplitude0) < 1e-12:
        amplitude0 = 1e-6

    # Keep exp(-g N) numerically bounded over the observed gate-count range.
    scale = max(float(np.max(np.abs(x))), 1.0)
    g_limit = 40.0 / scale

    try:
        params, _ = curve_fit(
            _exponential_model,
            x,
            y,
            p0=[amplitude0, 0.0],
            bounds=([-np.inf, -g_limit], [np.inf, g_limit]),
            maxfev=50000,
        )
        amplitude, decay_rate = (float(params[0]), float(params[1]))
        fitted = _exponential_model(x, amplitude, decay_rate)
        rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
        return {
            "success": True,
            "model": "A*exp(-g*N)",
            "amplitude": amplitude,
            "decay_rate": decay_rate,
            "at_N_zero": amplitude,
            "rmse": rmse,
        }
    except Exception as error:
        return {
            "success": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _build_axis_case(case_name, N_definition, N_values, expectations):
    N_values = [float(value) for value in N_values]
    return {
        "case": case_name,
        "N_definition": N_definition,
        "N": N_values,
        "linear": _linear_fit(N_values, expectations),
        "exponential": _exponential_fit(N_values, expectations),
    }


def _build_gate_axis_report(raw_counts, transpiled_counts, expectations):
    raw_1q = [row["N_1q"] for row in raw_counts]
    raw_2q = [row["N_2q"] for row in raw_counts]
    transpiled_1q = [row["N_1q"] for row in transpiled_counts]
    transpiled_2q = [row["N_2q"] for row in transpiled_counts]

    raw_weighted = [
        GATE_WEIGHT_1Q * n1 + GATE_WEIGHT_2Q * n2
        for n1, n2 in zip(raw_1q, raw_2q)
    ]
    transpiled_weighted = [
        GATE_WEIGHT_1Q * n1 + GATE_WEIGHT_2Q * n2
        for n1, n2 in zip(transpiled_1q, transpiled_2q)
    ]

    weighted_definition = (
        f"N = {GATE_WEIGHT_1Q}*N_1q + {GATE_WEIGHT_2Q}*N_2q"
    )

    cases = {
        "before_weighted": _build_axis_case(
            "before_transpilation_weighted",
            weighted_definition,
            raw_weighted,
            expectations,
        ),
        "before_two_qubit": _build_axis_case(
            "before_transpilation_two_qubit",
            "N = N_2q",
            raw_2q,
            expectations,
        ),
        "after_weighted": _build_axis_case(
            "after_transpilation_weighted",
            weighted_definition,
            transpiled_weighted,
            expectations,
        ),
        "after_two_qubit": _build_axis_case(
            "after_transpilation_two_qubit",
            "N = N_2q",
            transpiled_2q,
            expectations,
        ),
    }

    return {
        "gate_counts": {
            "before_transpilation": {
                "N_1q": raw_1q,
                "N_2q": raw_2q,
                "N_weighted": [float(v) for v in raw_weighted],
            },
            "after_transpilation": {
                "N_1q": transpiled_1q,
                "N_2q": transpiled_2q,
                "N_weighted": [float(v) for v in transpiled_weighted],
            },
        },
        "cases": cases,
    }


def _print_gate_axis_results(axis_report, precision):
    print("\nGate-count-axis extrapolation to N = 0")
    order = [
        "before_weighted",
        "before_two_qubit",
        "after_weighted",
        "after_two_qubit",
    ]
    for key in order:
        case = axis_report["cases"][key]
        print(f"\n{key}: {case['N_definition']}")
        print("N = [" + ", ".join(f"{value:.6g}" for value in case["N"]) + "]")
        for fit_name in ("linear", "exponential"):
            fit = case[fit_name]
            if fit.get("success"):
                print(
                    f"{fit_name}: {fit['at_N_zero']:.{precision}f} at N=0 "
                    f"(RMSE={fit['rmse']:.3e})"
                )
            else:
                print(f"{fit_name}: FIT FAILED - {fit.get('error', 'unknown error')}")


def _safe_filename_tag(value):
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


def _real_hardware_backend_name():
    if not USE_REAL_HARDWARE:
        return None
    name = backend.name
    if callable(name):
        name = name()
    return str(name)


def _backend_mode_name():
    if USE_REAL_HARDWARE:
        return f"real_hardware:{_real_hardware_backend_name()}"
    if USE_STATEVECTOR:
        return "statevector_noiseless"
    if USE_FAKE_BACKEND:
        return f"fake_backend:{FAKE_BACKEND_NAME}"
    if _INCREASING_NOISE_ACTIVE:
        return "aer_depth_dependent_noise"
    if _CONSTANT_NOISE_ACTIVE:
        return "aer_artificial_noise"
    return "aer_shot_noiseless"


def _backend_kind_for_report():
    if USE_REAL_HARDWARE:
        return "real_hardware"
    if USE_STATEVECTOR:
        return "statevector"
    if USE_FAKE_BACKEND:
        return "fake_backend"
    return "aer_simulator"


def _backend_name_for_report():
    if USE_REAL_HARDWARE:
        return _real_hardware_backend_name()
    if USE_STATEVECTOR:
        return "AerSimulator(statevector)"
    if USE_FAKE_BACKEND:
        return str(FAKE_BACKEND_NAME)
    return "AerSimulator"


def _machine_name_for_report():
    return "real_hardware" if USE_REAL_HARDWARE else str(SIMULATION_DEVICE)


def _noise_filename_tag():
    """Compact noise label used to prevent report-file overwrites."""
    if USE_FAKE_BACKEND:
        return None
    if USE_STATEVECTOR:
        return None
    if USE_REAL_HARDWARE:
        return None

    if USE_INCREASING_RANDOM_PAULI and USE_INCREASING_AMPLITUDE_DAMPING:
        return "increasing_pauli_amplitude_damping"
    if USE_INCREASING_RANDOM_PAULI:
        return "increasing_pauli"
    if USE_INCREASING_AMPLITUDE_DAMPING:
        return "increasing_amplitude_damping"

    # Preserve the historical 0.2% report naming so existing reports are
    # recognized and skipped by the batch runner.
    if (
        USE_CONSTANT_DEPOLARIZING
        and not USE_CONSTANT_AMPLITUDE_DAMPING
        and not USE_CONSTANT_PHASE_DAMPING
        and not USE_CONSTANT_COHERENT_OVERROTATION
        and abs(depolarizing_strength - 0.002) < 1e-12
    ):
        return "small_constant_depolarizing_0p2pct"

    constant = []
    if USE_CONSTANT_DEPOLARIZING:
        constant.append("depolarizing")
    if USE_CONSTANT_AMPLITUDE_DAMPING:
        constant.append("amplitude_damping")
    if USE_CONSTANT_PHASE_DAMPING:
        constant.append("phase_damping")
    if USE_CONSTANT_COHERENT_OVERROTATION:
        constant.append("coherent_overrotation")
    if constant:
        return "constant_" + "_".join(constant)
    return "noiseless"


def _execution_filename_tag():
    """Unique machine/backend/noise tag for report and plot filenames."""
    if USE_REAL_HARDWARE:
        return _safe_filename_tag(_real_hardware_backend_name())

    parts = [str(SIMULATION_DEVICE)]
    if USE_STATEVECTOR:
        parts.append("statevector")
    elif USE_FAKE_BACKEND:
        parts.append(_safe_filename_tag(FAKE_BACKEND_NAME))
    else:
        parts.append(_noise_filename_tag())
    return "_".join(part for part in parts if part)


def _noise_report():
    active_constant_types = []
    if USE_AER_SIMULATOR and USE_CONSTANT_DEPOLARIZING:
        active_constant_types.append("constant_depolarizing")
    if USE_AER_SIMULATOR and USE_CONSTANT_AMPLITUDE_DAMPING:
        active_constant_types.append("constant_amplitude_damping")
    if USE_AER_SIMULATOR and USE_CONSTANT_PHASE_DAMPING:
        active_constant_types.append("constant_phase_damping")
    if USE_AER_SIMULATOR and USE_CONSTANT_COHERENT_OVERROTATION:
        active_constant_types.append("constant_coherent_overrotation")

    active_increasing_types = []
    if USE_AER_SIMULATOR and USE_INCREASING_RANDOM_PAULI:
        active_increasing_types.append("depth_dependent_random_pauli")
    if USE_AER_SIMULATOR and USE_INCREASING_AMPLITUDE_DAMPING:
        active_increasing_types.append("depth_dependent_amplitude_damping")

    active_custom_types = active_constant_types + active_increasing_types

    if USE_FAKE_BACKEND:
        source = "fake_backend_device_snapshot"
    elif USE_REAL_HARDWARE:
        source = "real_hardware_device_noise"
    elif USE_STATEVECTOR:
        source = "none_statevector_noiseless"
    elif active_increasing_types:
        source = "depth_dependent_artificial_noise_channels"
    elif active_constant_types:
        source = "constant_artificial_noise_channels"
    else:
        source = "none_aer_shot_noiseless"

    return {
        "source": source,
        "selected_custom_noise_family": (
            "depth_dependent" if active_increasing_types
            else "constant" if active_constant_types
            else None
        ),
        "custom_noise_channels_applied": bool(active_custom_types),
        # Kept for compatibility with earlier reports.
        "active_custom_noise_types": active_custom_types,
        "active_constant_noise_types": active_constant_types,
        "active_depth_dependent_noise_types": active_increasing_types,
        "constant_noise_parameters": {
            "depolarizing_strength": float(depolarizing_strength),
            "amplitude_damping_strength": float(amplitude_damping_strength),
            "phase_damping_strength": float(phase_damping_strength),
            "coherent_overrotation_epsilon": float(overrotation_epsilon),
        },
        # Kept for compatibility with earlier reports.
        "custom_noise_parameters": {
            "depolarizing_strength": float(depolarizing_strength),
            "amplitude_damping_strength": float(amplitude_damping_strength),
            "phase_damping_strength": float(phase_damping_strength),
            "coherent_overrotation_epsilon": float(overrotation_epsilon),
        },
        "depth_dependent_noise_configuration": (
            NOISE_CONTROLLER.runtime_report() if _INCREASING_NOISE_ACTIVE else None
        ),
        "fake_backend_name": str(FAKE_BACKEND_NAME) if USE_FAKE_BACKEND else None,
    }


def _write_report(method_name, expectations, raw_counts, transpiled_counts,
                  ideal_expectation=None):
    axis_report = _build_gate_axis_report(
        raw_counts,
        transpiled_counts,
        expectations,
    )

    report = {
        "schema_version": 4,
        "method": method_name,
        "num_qubits": int(NUM_U_QUBITS),
        "unitary_type": str(UNITARY_TYPE),
        "depth": int(DEPTH),
        "fold_factors": [int(value) for value in depth_folded_circuits],
        "expectation_values": [float(value) for value in expectations],
        "ideal_expectation": (
            None if ideal_expectation is None else float(ideal_expectation)
        ),
        "backend_mode": _backend_mode_name(),
        "execution": {
            "machine": _machine_name_for_report(),
            "simulation_device": (
                None if USE_REAL_HARDWARE else str(SIMULATION_DEVICE)
            ),
            "gpu_enabled": bool(USE_GPU and not USE_REAL_HARDWARE),
            "backend_kind": _backend_kind_for_report(),
            "backend_mode": _backend_mode_name(),
            "backend_name": _backend_name_for_report(),
            "filename_tag": _execution_filename_tag(),
            "noise_filename_tag": _noise_filename_tag(),
            "shots": None if USE_STATEVECTOR else int(shots),
        },
        "noise": _noise_report(),
        "weights": {
            "w1_one_qubit": float(GATE_WEIGHT_1Q),
            "w2_two_qubit": float(GATE_WEIGHT_2Q),
        },
        "gate_count_definition": (
            "Structural gate count. Measurements, resets, barriers, delays, "
            "Aer save/set instructions, and explicit artificial-noise-channel "
            "instructions are excluded. Control-flow blocks are counted recursively."
        ),
        **axis_report,
    }

    output_directory = Path(REPORT_DIRECTORY).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    execution_tag = _execution_filename_tag()

    filename = (
        f"{NUM_U_QUBITS}q_{UNITARY_TYPE}_{DEPTH}_"
        f"{execution_tag}_{method_name}.json"
    )
    path = output_directory / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=False)

    return path, axis_report

# ============================================================
# Backend
# ============================================================


if USE_REAL_HARDWARE:

    if not QiskitRuntimeService.saved_accounts():
        raise RuntimeError(
            "USE_REAL_HARDWARE=True, but no IBM Quantum account is saved."
        )

    service = QiskitRuntimeService(channel="ibm_quantum")

    if REAL_HARDWARE_DEVICE_NAME:
        backend = service.backend(REAL_HARDWARE_DEVICE_NAME)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            dynamic_circuits=True,  # Keep behavior consistent with depth-reduced methods.
        )

elif USE_STATEVECTOR:

    backend = AerSimulator(
        method="statevector",
        precision="double",
        zero_threshold=0.0,
        max_parallel_experiments=1,
    )
    configure_aer_simulator_device(backend, use_gpu=USE_GPU)

elif USE_FAKE_BACKEND:

    FAKE_DEVICE, backend = make_fake_aer_backend(
        FAKE_BACKEND_NAME,
        seed_simulator=BASE_SEED,
        required_qubits=NUM_U_QUBITS,
        use_gpu=USE_GPU,
    )

elif USE_AER_SIMULATOR:

    backend = AerSimulator()
    configure_aer_simulator_device(backend, use_gpu=USE_GPU)
    backend.set_options(seed_simulator=BASE_SEED)

else:  # Protected by the machine/backend validation above.
    raise AssertionError("No execution backend was selected.")


# ============================================================
# Circuit running
# ============================================================

folded_circuits = []
COMPILATION_TOTALS = {
    "global_y_pairs": 0,
    "h_pairs": 0,
    "basis_rx_pairs": 0,
}

if DRAW_CIRCUIT:
    depths_to_build = [DRAW_DEPTH_FOLDED]
else:
    depths_to_build = depth_folded_circuits

for depth_folded in depths_to_build:
    q = QuantumRegister(NUM_U_QUBITS, 'q')
    c = ClassicalRegister(NUM_U_QUBITS, 'c')
    qc = QuantumCircuit(q, c)

    # ========================================================
    # 1. Prepare initial n-qubit state
    # ========================================================

    prepare_initial_state(
        qc,
        q
    )

    qc.barrier()

    # ========================================================
    # 2. Circuit folding
    #
    # depth_folded = 1:
    #     U
    #
    # depth_folded = 3:
    #     U -> Udagger -> U
    #
    # depth_folded = 5:
    #     U -> Udagger -> U -> Udagger -> U
    # ========================================================

    num_fold = (depth_folded - 1) // 2

    compile_stats = append_unitary_block(
        qc,
        q,
        SELECTED_COMPONENTS,
        inverse=False,
        dress_cx=False,
        noise_controller=NOISE_CONTROLLER,
        use_noise=(
            USE_AER_SIMULATOR
            and _CUSTOM_NOISE_ACTIVE
            and not DRAW_CIRCUIT
            and not USE_STATEVECTOR
        ),
        compile_cancellations=True,
    )
    for key, value in compile_stats.items():
        COMPILATION_TOTALS[key] += value

    for _ in range(num_fold):
        compile_stats = append_unitary_block(
            qc,
            q,
            SELECTED_COMPONENTS,
            inverse=True,
            dress_cx=False,
            noise_controller=NOISE_CONTROLLER,
            use_noise=(
                USE_AER_SIMULATOR
                and _CUSTOM_NOISE_ACTIVE
                and not DRAW_CIRCUIT
                and not USE_STATEVECTOR
            ),
            compile_cancellations=True,
        )
        for key, value in compile_stats.items():
            COMPILATION_TOTALS[key] += value

        compile_stats = append_unitary_block(
            qc,
            q,
            SELECTED_COMPONENTS,
            inverse=False,
            dress_cx=False,
            noise_controller=NOISE_CONTROLLER,
            use_noise=(
                USE_AER_SIMULATOR
                and _CUSTOM_NOISE_ACTIVE
                and not DRAW_CIRCUIT
                and not USE_STATEVECTOR
            ),
            compile_cancellations=True,
        )
        for key, value in compile_stats.items():
            COMPILATION_TOTALS[key] += value


    # ========================================================
    # 3. Final measurement
    # ========================================================

    for k in range(NUM_U_QUBITS):

        qc.measure(
            q[k],
            c[k]
        )

    folded_circuits.append(qc)

    # ========================================================
    # Draw circuit only
    # ========================================================

    if DRAW_CIRCUIT:

        print(
            f"\nDepth_folded = {depth_folded}"
        )

        qc.draw(
            'mpl',
            style='clifford'
        )

        plt.show()

        raise SystemExit

# ============================================================
# Select readout for execution
# ============================================================
if USE_STATEVECTOR:
    simulation_circuits = []
    observable = Pauli("Z" * NUM_U_QUBITS)

    for circuit in folded_circuits:
        qc_exact = circuit.remove_final_measurements(inplace=False)
        if any(item.operation.name == "measure" for item in qc_exact.data):
            raise ValueError(
                "USE_STATEVECTOR requires a pure-state preparation without "
                "intermediate measurements."
            )
        qc_exact.save_expectation_value(
            observable,
            list(range(NUM_U_QUBITS)),
            label="expectation_value",
        )
        simulation_circuits.append(qc_exact)
else:
    simulation_circuits = folded_circuits

# ============================================================
# Transpilation
# ============================================================
pm = generate_preset_pass_manager(
    backend=backend,
    optimization_level=0
)

exec_circuits = [
    pm.run(circuit)
    for circuit in simulation_circuits
]

# ============================================================
# Run every folded circuit and collect expectations
# ============================================================
if USE_STATEVECTOR:
    # One deterministic state evolution per circuit. There are no noise
    job = backend.run(exec_circuits, shots=1)
    result = job.result()
    if not result.success:
        raise RuntimeError(f"Statevector simulation failed: {result.status}")

    expectation_values = [
        float(result.data(i)["expectation_value"])
        for i in range(len(exec_circuits))
    ]
else:
    sampler = Sampler(mode=backend)

    job = sampler.run(
        exec_circuits,
        shots=shots
    )

    result = job.result()

    # ============================================================
    # Z expectation value for n-qubit output
    #
    # <Z^{\otimes n}> = sum_x (-1)^{|x|} P(x)
    # ============================================================

    expectation_values = []

    for i in range(len(folded_circuits)):

        counts = result[i].join_data().get_counts()

        total_counts = sum(
            counts.values()
        )

        expectation = 0.0

        for bitstring, count in counts.items():

            bitstring = bitstring.replace(
                " ",
                ""
            )

            eigenvalue = (
                (-1) ** bitstring.count("1")
            )

            expectation += (
                eigenvalue
                * count
                / total_counts
            )

        expectation_values.append(
            expectation
        )


# ============================================================
# Print experiment configuration
# ============================================================

print(f"\nUNITARY_TYPE = {UNITARY_TYPE}")
print(selected_unitary_summary(UNITARY_TYPE, NUM_U_QUBITS, DEPTH))
print(f"DEPTH = {DEPTH}; maximum configured depth = {MAX_U_DEPTH}")
print(
    "One-pass U compilation removed pairs: "
    f"H={COMPILATION_TOTALS['h_pairs']}, "
    f"inverse RX(pi/2)={COMPILATION_TOTALS['basis_rx_pairs']}"
)

if USE_REAL_HARDWARE:
    print("Machine: real quantum hardware")
    print(f"Backend: {_real_hardware_backend_name()}")
else:
    print(f"Machine: {SIMULATION_DEVICE}")

    if USE_STATEVECTOR:
        print("Backend: AerSimulator (statevector, noiseless)")
    elif USE_FAKE_BACKEND:
        print(f"Backend: {FAKE_BACKEND_NAME} snapshot via AerSimulator")
        print("Noise: fake-backend device snapshot; custom noise channels disabled")
    elif USE_AER_SIMULATOR:
        print("Backend: AerSimulator")

        active_noises = []
        if USE_CONSTANT_DEPOLARIZING:
            active_noises.append("constant depolarizing")
        if USE_CONSTANT_AMPLITUDE_DAMPING:
            active_noises.append("constant amplitude damping")
        if USE_CONSTANT_PHASE_DAMPING:
            active_noises.append("constant phase damping")
        if USE_CONSTANT_COHERENT_OVERROTATION:
            active_noises.append("constant coherent overrotation")
        if USE_INCREASING_RANDOM_PAULI:
            active_noises.append("depth-dependent random Pauli")
        if USE_INCREASING_AMPLITUDE_DAMPING:
            active_noises.append("depth-dependent amplitude damping")

        print("Noise:", ", ".join(active_noises) if active_noises else "none")
    else:
        raise AssertionError("Unexpected simulator-backend selection while printing configuration.")

precision = 12 if USE_STATEVECTOR else 4
print(f"\nExpectation values of circuit_folded method:")
print("[" + ", ".join(f"{x:.{precision}f}" for x in expectation_values) + "]")

# ============================================================
# Gate counts before/after transpilation
# ============================================================
raw_gate_counts = _gate_count_list(folded_circuits)
transpiled_gate_counts = _gate_count_list(exec_circuits)

# ============================================================
# Exact ideal value for noisy Aer or fake-backend runs
# ============================================================
# This is intentionally done only for standard circuit folding, as requested.
# It does not reuse the noisy circuit because that circuit contains explicit
# channel instructions.  Instead it rebuilds the same fold-factor-1 U with
# use_noise=False and evaluates <Z^{\otimes n}> by a noiseless statevector.
_need_ideal_reference = (
    not USE_REAL_HARDWARE
    and not USE_STATEVECTOR
    and (USE_FAKE_BACKEND or (USE_AER_SIMULATOR and _CUSTOM_NOISE_ACTIVE))
)

ideal_expectation = None

if _need_ideal_reference:
    ideal_backend = AerSimulator(
        method="statevector",
        precision="double",
        zero_threshold=0.0,
        max_parallel_experiments=1,
    )
    configure_aer_simulator_device(ideal_backend, use_gpu=USE_GPU)

    q_ideal = QuantumRegister(NUM_U_QUBITS, "q_ideal")
    ideal_circuit = QuantumCircuit(q_ideal)
    prepare_initial_state(ideal_circuit, q_ideal)
    ideal_circuit.barrier()
    append_unitary_block(
        ideal_circuit,
        q_ideal,
        SELECTED_COMPONENTS,
        inverse=False,
        dress_cx=False,
        noise_controller=NOISE_CONTROLLER,
        use_noise=False,
        compile_cancellations=True,
    )
    ideal_circuit.save_expectation_value(
        Pauli("Z" * NUM_U_QUBITS),
        list(range(NUM_U_QUBITS)),
        label="ideal_expectation",
    )
    ideal_pm = generate_preset_pass_manager(
        backend=ideal_backend,
        optimization_level=0,
    )
    ideal_exec = ideal_pm.run(ideal_circuit)
    ideal_result = ideal_backend.run(ideal_exec, shots=1).result()
    if not ideal_result.success:
        raise RuntimeError(
            f"Noiseless ideal-statevector calculation failed: {ideal_result.status}"
        )
    ideal_expectation = float(
        np.real(ideal_result.data(0)["ideal_expectation"])
    )
    print(f"Ideal statevector expectation: {ideal_expectation:.12f}")

# ============================================================
# Linear and exponential extrapolation on four gate-count axes
# ============================================================
report_path, axis_report = _write_report(
    "circuit_folding",
    expectation_values,
    raw_gate_counts,
    transpiled_gate_counts,
    ideal_expectation=ideal_expectation,
)
_print_gate_axis_results(axis_report, precision)
print(f"\nReport saved to: {report_path}")
