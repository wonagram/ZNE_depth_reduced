from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
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
    append_local_component_copy,
    configure_aer_simulator_device,
    make_fake_aer_backend,
    make_fake_statevector_aer_backend,
    selected_component_matrix,
)
from state_preparation import *

from zne_cli import (
    override,
    parse_cli_args,
    resolve_backend_flags,
    resolve_machine_flags,
    resolve_noise_flags,
)

_CLI_ARGS = parse_cli_args('local depth-reduced ZNE experiment')

from local_folding_helpers import (
    make_local_plan,
    build_intermediate_local_circuit,
    check_intermediate_local_folding,
    describe_local_plan,
)

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
# Select exactly one of the four simulation backends.
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
    "USE_FAKE_STATEVECTOR": USE_FAKE_STATEVECTOR,
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
            "USE_STATEVECTOR, USE_FAKE_BACKEND, and USE_FAKE_STATEVECTOR. Enabled simulation backends: "
            + ", ".join(_ENABLED_SIMULATION_BACKENDS)
        )
    SIMULATION_DEVICE = None
else:
    if len(_ENABLED_SIMULATION_BACKENDS) != 1:
        raise ValueError(
            "When USE_REAL_HARDWARE=False, select exactly one simulation backend: "
            "USE_AER_SIMULATOR, USE_STATEVECTOR, USE_FAKE_BACKEND, or USE_FAKE_STATEVECTOR. Enabled: "
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

# Physical pool = NUM_U_QUBITS * fold factor, independent of DEPTH.
# "fixed": permanent lanes. "dynamic": shared free/reset workspace.
QUBIT_ASSIGNMENT = "dynamic"

# "front": input state on q0,...,q(n-1).
# "first_use": spatial placement follows first component use.
INPUT_PLACEMENT = "first_use"

# "snake": alternating directions. "cycle": restart from the top.
REUSE_PATTERN = "cycle"

# ============================================================
# Circuit drawing
# ============================================================
DRAW_CIRCUIT = False
DRAW_DEPTH_FOLDED = 3

# ============================================================
# Exact, noiseless branch checking
# ============================================================
CHECK_CODE = False
CHECK_DEPTH_FOLDED = 3
CHECK_MAX_PHYSICAL_QUBITS = 20
CHECK_MAX_BRANCHES = 4096
CHECK_PRINT_EACH = True

# Guard against accidentally constructing an impractically wide simulator job.
MAX_SIMULATOR_PHYSICAL_QUBITS = int(override(
    24,
    _CLI_ARGS.max_simulator_physical_qubits,
))

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
depolarizing_strength = 0.02
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
# Select the exact PHYSICAL-depth prefix.
# IMPORTANT FOR LOCAL FOLDING:
# A full or cut CZ/CRZ/CRX/RZZ/RXX/RYY macro remains ONE atomic G_i.
# Its decomposition slices determine DEPTH, but they are never separate local-folding G_i.
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
    if USE_FAKE_STATEVECTOR:
        return f"fake_backend_statevector_trajectories:{FAKE_BACKEND_NAME}"
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
    if USE_FAKE_STATEVECTOR:
        return "fake_statevector"
    if USE_FAKE_BACKEND:
        return "fake_backend"
    return "aer_simulator"


def _backend_name_for_report():
    if USE_REAL_HARDWARE:
        return _real_hardware_backend_name()
    if USE_STATEVECTOR:
        return "AerSimulator(statevector)"
    if USE_FAKE_STATEVECTOR:
        return f"{FAKE_BACKEND_NAME}(statevector trajectories)"
    if USE_FAKE_BACKEND:
        return str(FAKE_BACKEND_NAME)
    return "AerSimulator"


def _machine_name_for_report():
    return "real_hardware" if USE_REAL_HARDWARE else str(SIMULATION_DEVICE)


def _noise_filename_tag():
    """Compact noise label used to prevent report-file overwrites."""
    if USE_FAKE_BACKEND or USE_FAKE_STATEVECTOR:
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
    elif USE_FAKE_STATEVECTOR:
        parts.append(_safe_filename_tag(FAKE_BACKEND_NAME))
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

    if USE_FAKE_STATEVECTOR:
        source = "fake_backend_device_snapshot_statevector_trajectories"
    elif USE_FAKE_BACKEND:
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
        "fake_backend_name": (
            str(FAKE_BACKEND_NAME)
            if (USE_FAKE_BACKEND or USE_FAKE_STATEVECTOR)
            else None
        ),
        "fake_statevector_trajectory_sampling": bool(USE_FAKE_STATEVECTOR),
        "shot_branching_enable": bool(USE_FAKE_STATEVECTOR),
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
            "simulation_method": (
                "statevector"
                if (USE_STATEVECTOR or USE_FAKE_STATEVECTOR)
                else "automatic"
            ),
            "noisy_statevector_trajectories": bool(USE_FAKE_STATEVECTOR),
            "shot_branching_enable": bool(USE_FAKE_STATEVECTOR),
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

def apply_selected_component(qc, mapped_wires, component, block_index, use_noise):
    """Apply one WHOLE logical macro to one copied block.

    Example: one CZ component emits H-CX-H here before the helper performs
    that component's Bell measurements.  RYY similarly emits all five
    physical-depth slices as one G_i.  A DEPTH-cut prefix is also one G_i.
    """
    append_local_component_copy(
        qc,
        mapped_wires,
        component,
        block_index=block_index,
        noise_controller=NOISE_CONTROLLER,
        use_noise=use_noise,
    )


def assert_atomic_local_plan(plan, selected_components):
    """Prevent decomposition slices from becoming separate local-folding stages."""
    stages = plan.get("stages", [])
    if len(stages) != len(selected_components):
        raise ValueError(
            "Local folding must create exactly one G_i per selected logical "
            f"component, but got {len(stages)} stages for "
            f"{len(selected_components)} components. Decomposition slices must "
            "not be planned as separate local-folding gates."
        )

    for index, (stage, expected) in enumerate(zip(stages, selected_components), start=1):
        actual = stage.get("component")
        if actual != expected:
            raise ValueError(
                f"G{index} does not contain the original selected macro component. "
                "Keep CZ/CRZ/CRX/RZZ/RXX/RYY decompositions atomic in local folding."
            )
        if getattr(actual, "included_depth", None) != getattr(expected, "included_depth", None):
            raise ValueError(f"G{index} lost the selected macro-prefix depth.")


def psi_minus(qc, wires):
    qc.h(wires[0])
    qc.cx(wires[0], wires[1])
    qc.z(wires[0])
    qc.x(wires[1])


def bell_measure(qc, wires):
    qc.cx(wires[0], wires[1])
    qc.h(wires[0])


def apply_correction(qc, c_pair, target):
    """
    Bell outcome:
        00 -> XZ
        01 -> Z
        10 -> X
        11 -> I
    """
    with qc.if_test((c_pair, 0)):
        qc.z(target)
        qc.x(target)
    with qc.if_test((c_pair, 1)):
        qc.z(target)
    with qc.if_test((c_pair, 2)):
        qc.x(target)

# ============================================================
# Noiseless statevector execution with intermediate qubit reuse
# ============================================================
# These functions read the ACTUAL circuit emitted by the existing helper.
# They do not replace a folded component with its ideal logical gate.
# Measurement records are kept coherently within each component. Before
# resetting/reusing their wires, we verify that they factor from ALL other
# wires. Only then is the record factor replaced by |0...0>.
# No density matrix, noise trajectories, or measurement samples are used.


def _sv_preparation(circuit, prep_end):
    """Convert a pure input preparation to a measurement-free circuit."""
    from qiskit.circuit import Gate

    result = QuantumCircuit(circuit.num_qubits)
    result.global_phase = circuit.global_phase
    touched = set()
    for item in circuit.data[:prep_end]:
        op = item.operation
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if op.name == "barrier":
            continue
        if item.clbits or getattr(op, "condition", None) is not None:
            raise ValueError("Statevector input preparation must not use classical bits.")
        if op.name in ("initialize", "reset"):
            # Initialize normally includes reset. It is safely unitary here
            # ONLY on previously untouched wires, which are known to be |0>.
            if touched.intersection(wires):
                raise ValueError(
                    "Statevector mode needs pure input preparation. initialize/reset "
                    "on previously used wires is not silently sampled. Use a unitary "
                    "preparation or initialize each input once on fresh |0> wires."
                )
            if op.name == "reset":
                continue
            prep_gate = op.gates_to_uncompute().inverse().to_gate()
            result.append(prep_gate, wires)
        elif isinstance(op, Gate):
            result.append(op, wires)
        else:
            raise ValueError(
                f"Unsupported input instruction {op.name!r} in statevector mode. "
                "Use unitary gates or initialize on untouched input wires."
            )
        touched.update(wires)
    return result


def _sv_coherent_component(circuit, record, stage):
    """Translate the component's actual measurement/if_test records coherently.

    The current helper uses independent (register, integer) if_test blocks
    containing Pauli gates and no else branches. Unsupported control flow is
    rejected rather than ignored. Measurement controls cannot be overwritten.
    """
    from qiskit.circuit import Gate

    width = circuit.num_qubits
    result = QuantumCircuit(width)
    measured = {}  # Actual classical-bit object -> physical measurement wire.
    correction_started = False
    start, stop = record["start"], record["corrections_end"]
    for item in circuit.data[start:stop]:
        op = item.operation
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if op.name == "barrier":
            continue
        if op.name == "measure":
            if correction_started or len(wires) != 1 or len(item.clbits) != 1:
                raise ValueError("A component must measure all Bell wires before correcting.")
            bit = item.clbits[0]
            if bit in measured or wires[0] in measured.values():
                raise ValueError("A Bell bit/wire is measured twice in the same component.")
            measured[bit] = wires[0]
            continue
        if op.name == "if_else":
            correction_started = True
            condition = op.condition
            if not isinstance(condition, tuple) or len(condition) != 2:
                raise ValueError("Statevector mode requires (classical bit/register, value) conditions.")
            classical, value = condition
            bits = list(classical) if isinstance(classical, ClassicalRegister) else [classical]
            if not bits or any(bit not in measured for bit in bits):
                raise ValueError("A correction depends on a bit outside this component's Bell records.")
            controls = [measured[bit] for bit in bits]
            value = int(value)
            if not 0 <= value < (1 << len(controls)):
                raise ValueError("Classical correction value is outside its register range.")
            if len(op.blocks) > 1 and op.blocks[1] is not None:
                raise ValueError("This statevector path supports if_test corrections without else.")
            body = op.blocks[0]
            if abs(float(body.global_phase)) > 1e-14:
                raise ValueError("A phase-bearing conditional block needs an explicit coherent translation.")
            if len(body.qubits) != len(wires):
                raise ValueError("Unexpected if_test qubit mapping.")
            for inner in body.data:
                gate = inner.operation
                if gate.name == "barrier":
                    continue
                targets = [wires[body.find_bit(q).index] for q in inner.qubits]
                if (inner.clbits or not isinstance(gate, Gate)
                        or getattr(gate, "condition", None) is not None):
                    raise ValueError("Only unitary correction gates are supported inside if_test.")
                if set(targets).intersection(measured.values()):
                    raise ValueError("A correction must not overwrite a Bell-record wire.")
                # Register iteration is little-endian. ctrl_state=value
                # therefore uses the original c[1]c[0] convention exactly.
                result.append(
                    gate.control(num_ctrl_qubits=len(controls), ctrl_state=value),
                    controls + targets,
                )
            continue
        if measured or correction_started:
            raise ValueError("Unexpected unconditional gate after the component's Bell measurements.")
        if item.clbits or not isinstance(op, Gate) or getattr(op, "condition", None) is not None:
            raise ValueError(f"Non-unitary/unsupported component instruction: {op.name!r}.")
        result.append(op, wires)

    actual_resets = []
    for item in circuit.data[stop:record["end"]]:
        if item.operation.name == "barrier":
            continue
        if item.operation.name != "reset" or item.clbits or len(item.qubits) != 1:
            raise ValueError("Only workspace resets may follow the component's corrections.")
        actual_resets.append(circuit.find_bit(item.qubits[0]).index)
    if (actual_resets != list(record["reset_ids"])
            or actual_resets != list(stage["reset_ids"])
            or len(set(actual_resets)) != len(actual_resets)
            or set(actual_resets) != set(measured.values())):
        raise ValueError("Actual Bell measurements/resets do not match the reuse plan.")
    if set(actual_resets).intersection(record["output_ids"]):
        raise ValueError("A reset would erase a live logical output/spectator.")
    return result, actual_resets


def _sv_run_segment(segment, backend, initial=None, output_ids=None):
    """Evolve a measurement-free segment using genuine Aer statevector APIs."""
    run = QuantumCircuit(segment.num_qubits)
    if initial is not None:
        run.set_statevector(initial)
    run.compose(segment, inplace=True)
    run.save_statevector(label="local_statevector")
    if output_ids is not None:
        run.save_expectation_value(
            Pauli("Z" * len(output_ids)), output_ids, label="local_expectation"
        )
    compiled = transpile(run, backend, optimization_level=0)
    if compiled.num_qubits != segment.num_qubits:
        raise ValueError("Transpilation changed the physical register width.")
    if compiled.layout is not None:
        if compiled.layout.final_index_layout() != list(range(segment.num_qubits)):
            raise ValueError("Unexpected simulator layout: statevector wire order changed.")
    forbidden = {"measure", "reset", "if_else", "kraus", "superop", "quantum_channel"}
    if any(item.operation.name in forbidden for item in compiled.data):
        raise ValueError("A stochastic instruction reached the no-shot statevector backend.")
    # Aer uses the shots argument as an execution count. There are NO
    # measurements, resets, or noise here, so this is not a one-shot estimate.
    execution = backend.run(compiled, shots=1).result()
    if not execution.success:
        raise RuntimeError(f"Aer statevector execution failed: {execution.status}")
    data = execution.data(0)
    vector = np.asarray(data["local_statevector"], dtype=np.complex128).copy()
    norm_squared = float(np.vdot(vector, vector).real)
    if not np.isfinite(norm_squared) or abs(norm_squared - 1.0) > 1e-9:
        raise ValueError(f"Invalid statevector norm squared: {norm_squared}.")
    expectation = None
    if output_ids is not None:
        value = complex(data["local_expectation"])
        if abs(value.imag) > 1e-10 or not np.isfinite(value.real):
            raise ValueError(f"Invalid real observable expectation: {value}.")
        expectation = float(value.real)
    return vector, expectation


def _sv_reset_factored_records(vector, reset_ids, width):
    """Discard a record factor ONLY after testing factorization of all amplitudes.

    Reshaping the vector into record/live indices uses 2**width amplitudes,
    NOT a 2**width by 2**width density matrix. A reference row identifies
    the candidate live vector, but EVERY row is checked, including phases.
    This is not postselection, branch sampling, or an unchecked approximation.
    """
    if not reset_ids:
        return vector, 0.0
    reset_ids = list(reset_ids)
    if len(set(reset_ids)) != len(reset_ids) or any(q < 0 or q >= width for q in reset_ids):
        raise ValueError("Invalid reset wire list.")
    keep_ids = [q for q in range(width) if q not in set(reset_ids)]
    axes = [width - 1 - q for q in reversed(reset_ids)]
    axes += [width - 1 - q for q in reversed(keep_ids)]
    grouped = vector.reshape((2,) * width).transpose(axes).reshape(1 << len(reset_ids), -1)
    row_norms = np.sum(np.abs(grouped) ** 2, axis=1)
    pivot = int(np.argmax(row_norms))
    norm_squared = float(np.sum(row_norms))
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ValueError("Invalid coherent state before workspace reset.")
    live = grouped[pivot].copy() / np.sqrt(row_norms[pivot])
    # Check all record sectors in bounded temporary buffers. The residual is
    # ||Psi - record_factor tensor live||_2 / ||Psi||_2.
    chunk_rows = max(1, (1 << 16) // grouped.shape[1])
    residual_squared = 0.0
    conjugate_live = live.conj()
    for start in range(0, grouped.shape[0], chunk_rows):
        block = grouped[start:start + chunk_rows]
        weights = block @ conjugate_live
        difference = block - weights[:, None] * live[None, :]
        residual_squared += float(np.vdot(difference, difference).real)
    residual = float(np.sqrt(max(0.0, residual_squared) / norm_squared))
    if residual > 1e-11:
        raise ValueError(
            f"Bell records remain correlated with live qubits (residual={residual:.3e}). "
            "Exact single-statevector reuse is not valid for this component. "
            "Check the circuit/corrections, use CHECK_CODE on a smaller case, "
            "or use USE_STATEVECTOR=False. No branch was sampled or discarded."
        )
    # All conditional live states agree up to phase to numerical precision.
    # Physical reset now replaces only the independent record factor by |0>.
    reset_grouped = np.zeros_like(grouped)
    reset_grouped[0] = live
    restored = reset_grouped.reshape((2,) * width).transpose(np.argsort(axes))
    return np.ascontiguousarray(restored).reshape(-1), residual


def local_statevector_expectation(circuit, plan, backend):
    """Run all actual folded components; return the no-shot output expectation.

    The metadata is produced by the user's existing local_folding_helpers.
    Rank-one record/live factorization is checked at EVERY reset boundary.
    Failure stops the run instead of returning a sampled or postselected value.
    """
    meta = (circuit.metadata or {}).get("intermediate_local_folding")
    required = {"prep_end", "stages", "output_start", "output_ids", "use_noise"}
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError("The local helper must provide intermediate_local_folding stage metadata.")
    if meta["use_noise"]:
        raise ValueError("Statevector mode requires use_noise=False when building the circuit.")
    if (circuit.num_qubits != plan["num_physical_qubits"]
            or len(meta["stages"]) != len(plan["stages"])
            or list(meta["output_ids"]) != list(plan["output_ids"])):
        raise ValueError("The circuit and local folding plan do not match.")
    out_registers = [register for register in circuit.cregs if register.name == "out"]
    if len(out_registers) != 1 or len(out_registers[0]) != plan["n"]:
        raise ValueError("Expected one final output register named out.")
    readout = list(circuit.data[meta["output_start"]:])
    if len(readout) != plan["n"]:
        raise ValueError("Unexpected operations after the final local component.")
    for k, item in enumerate(readout):
        if (item.operation.name != "measure" or len(item.qubits) != 1
                or len(item.clbits) != 1
                or circuit.find_bit(item.qubits[0]).index != plan["output_ids"][k]
                or item.clbits[0] != out_registers[0][k]):
            raise ValueError("The final readout does not match the logical output map.")

    previous_end = int(meta["prep_end"])
    if not 0 <= previous_end <= meta["output_start"] <= len(circuit.data):
        raise ValueError("Invalid local component boundaries.")
    preparation = _sv_preparation(circuit, previous_end)
    vector, expectation = _sv_run_segment(
        preparation, backend, output_ids=list(plan["input_ids"])
    )
    max_residual = 0.0
    for j, (record, stage) in enumerate(zip(meta["stages"], plan["stages"])):
        needed = {"layer", "start", "end", "corrections_end", "output_ids", "reset_ids"}
        if not needed.issubset(record):
            raise ValueError("The local helper's stage metadata is incomplete.")
        if (record["layer"] != j or record["start"] != previous_end
                or not record["start"] <= record["corrections_end"] <= record["end"] <= meta["output_start"]
                or list(record["output_ids"]) != list(stage["outputs_after_layer"])):
            raise ValueError(f"G{j + 1}: inconsistent component/output metadata.")
        coherent, reset_ids = _sv_coherent_component(circuit, record, stage)
        vector, expectation = _sv_run_segment(
            coherent, backend, initial=vector, output_ids=list(record["output_ids"])
        )
        try:
            vector, residual = _sv_reset_factored_records(vector, reset_ids, circuit.num_qubits)
        except ValueError as error:
            raise ValueError(f"fold={plan['fold_factor']}, G{j + 1}: {error}") from error
        max_residual = max(max_residual, residual)
        previous_end = record["end"]
    if previous_end != meta["output_start"]:
        raise ValueError("Some circuit operations were not included in the local components.")
    return expectation, max_residual


folded_circuits = []
local_plans = []

if CHECK_CODE:
    depths_to_build = [CHECK_DEPTH_FOLDED]
elif DRAW_CIRCUIT:
    depths_to_build = [DRAW_DEPTH_FOLDED]
else:
    depths_to_build = depth_folded_circuits


for depth_folded in depths_to_build:

    # --------------------------------------------------------
    # 1. Preserve the user-defined component/layer boundaries.
    #    Reuse reset wires with the chosen allocation/routing policies.
    # --------------------------------------------------------
    plan = make_local_plan(
        NUM_U_QUBITS,
        SELECTED_COMPONENTS,
        depth_folded,
        reuse_pattern=REUSE_PATTERN,
        qubit_assignment=QUBIT_ASSIGNMENT,
        input_placement=INPUT_PLACEMENT,
    )
    assert_atomic_local_plan(plan, SELECTED_COMPONENTS)
    describe_local_plan(plan)

    if (
        not USE_REAL_HARDWARE
        and not DRAW_CIRCUIT
        and not CHECK_CODE
        and plan["num_physical_qubits"] > MAX_SIMULATOR_PHYSICAL_QUBITS
    ):
        raise ValueError(
            f"Local circuit uses {plan['num_physical_qubits']} physical qubits. "
            "Reduce NUM_U_QUBITS/fold factors or deliberately increase "
            "MAX_SIMULATOR_PHYSICAL_QUBITS after checking resources."
        )

    # --------------------------------------------------------
    # 2. For EACH G_i:
    #    copies -> all its Bell measurements -> corrections -> reset measured wires.
    #    Only after correcting do we append G_(i+1).
    #
    # The returned circuit ALSO includes the final out readout.
    # Do not call the old finish_local_circuit() on it.
    # --------------------------------------------------------
    qc = build_intermediate_local_circuit(
        plan,
        prepare_initial_state=prepare_initial_state,
        apply_component=apply_selected_component,
        psi_minus=psi_minus,
        bell_measure=bell_measure,
        apply_correction=apply_correction,
        use_noise=(
            USE_AER_SIMULATOR
            and _CUSTOM_NOISE_ACTIVE
            and not DRAW_CIRCUIT
            and not CHECK_CODE
            and not USE_STATEVECTOR
        ),
        stage_barriers=True,
    )

    # --------------------------------------------------------
    # 3. Exact branch-by-branch dynamic-circuit check.
    #    The checker projects each intermediate measurement,
    #    follows actual if_test corrections and resets before reusing wires.
    #    No backend is selected and no circuit job is submitted.
    # --------------------------------------------------------
    if CHECK_CODE:
        check_report = check_intermediate_local_folding(
            qc,
            plan,
            prepare_initial_state=prepare_initial_state,
            component_matrix=selected_component_matrix,
            max_physical_qubits=CHECK_MAX_PHYSICAL_QUBITS,
            max_branches=CHECK_MAX_BRANCHES,
            print_each=CHECK_PRINT_EACH,
        )
        raise SystemExit(0 if check_report["passed"] else 1)

    # --------------------------------------------------------
    # 4. Draw the FULL circuit, including intermediate
    #    conditional Pauli corrections and final readout.
    # --------------------------------------------------------
    if DRAW_CIRCUIT:
        print("\nFull intermediate-measurement/correction circuit.")
        qc.draw(
            "mpl",
            style="clifford",
            plot_barriers=True,
            fold=-1
        )
        plt.show()
        raise SystemExit

    folded_circuits.append(qc)
    local_plans.append(plan)


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
            dynamic_circuits=True,
        )

elif USE_STATEVECTOR:
    backend = AerSimulator(
        method="statevector",
        precision="double",
        zero_threshold=0.0,
        enable_truncation=False,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    configure_aer_simulator_device(backend, use_gpu=USE_GPU)

elif USE_FAKE_STATEVECTOR:

    FAKE_DEVICE, backend = make_fake_statevector_aer_backend(
        FAKE_BACKEND_NAME,
        seed_simulator=BASE_SEED,
        required_qubits=NUM_U_QUBITS * max(depth_folded_circuits),
        use_gpu=USE_GPU,
        shot_branching_enable=True,
    )

elif USE_FAKE_BACKEND:

    FAKE_DEVICE, backend = make_fake_aer_backend(
        FAKE_BACKEND_NAME,
        seed_simulator=BASE_SEED,
        required_qubits=NUM_U_QUBITS * max(depth_folded_circuits),
        use_gpu=USE_GPU,
    )

elif USE_AER_SIMULATOR:

    backend = AerSimulator()
    configure_aer_simulator_device(backend, use_gpu=USE_GPU)
    backend.set_options(seed_simulator=BASE_SEED)

else:  # Protected by the machine/backend validation above.
    raise AssertionError("No execution backend was selected.")


# ============================================================
# Transpilation
# ============================================================
if USE_STATEVECTOR:
    expectation_values = []
    for circuit, plan in zip(folded_circuits, local_plans):
        expectation, reset_residual = local_statevector_expectation(circuit, plan, backend)
        expectation_values.append(expectation)
        print(
            f"Statevector fold={plan['fold_factor']}: {expectation:.12f}; "
            f"max reset-factorization residual={reset_residual:.3e}"
        )

    # The exact local statevector path executes segments rather than the full
    # dynamic circuit.  Transpile the unchanged full circuit only to obtain the
    # requested after-transpilation structural gate counts for the report.
    count_pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=0,
    )
    exec_circuits = [
        count_pm.run(circuit)
        for circuit in folded_circuits
    ]
else:
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=0
    )

    exec_circuits = [
        pm.run(circuit)
        for circuit in folded_circuits
    ]

    sampler = Sampler(mode=backend)

    job = sampler.run(
        exec_circuits,
        shots=shots
    )

    result = job.result()

    # ============================================================
    # Z expectation value for n-qubit output
    #
    # <Z^{\otimes n}> = sum_{x in {0,1}^n} (-1)^{|x|} P(x)
    #
    # where |x| is the number of 1s in the bitstring x.
    # Therefore:
    #   even number of 1s -> +1
    #   odd  number of 1s -> -1
    # ============================================================

    expectation_values = []

    for i in range(len(folded_circuits)):

        out_counts = result[i].data.out.get_counts()

        total_counts = sum(out_counts.values())

        expectation = 0.0

        for bitstring, count in out_counts.items():

            # Remove spaces if present
            bitstring = bitstring.replace(" ", "")

            # Eigenvalue of Z^{\otimes n}:
            # even number of 1s -> +1
            # odd  number of 1s -> -1
            eigenvalue = (-1) ** bitstring.count("1")

            expectation += eigenvalue * count / total_counts

        expectation_values.append(expectation)


# ============================================================
# Print experiment configuration
# ============================================================

print(f"\nUNITARY_TYPE = {UNITARY_TYPE}")
print(selected_unitary_summary(UNITARY_TYPE, NUM_U_QUBITS, DEPTH))
print(f"DEPTH = {DEPTH}; maximum configured depth = {MAX_U_DEPTH}")

if USE_REAL_HARDWARE:
    print("Machine: real quantum hardware")
    print(f"Backend: {_real_hardware_backend_name()}")
else:
    print(f"Machine: {SIMULATION_DEVICE}")

    if USE_STATEVECTOR:
        print("Backend: AerSimulator (statevector, noiseless)")
    elif USE_FAKE_STATEVECTOR:
        print(
            f"Backend: {FAKE_BACKEND_NAME} snapshot via AerSimulator "
            "(forced statevector trajectories)"
        )
        print(
            "Noise: fake-backend device snapshot; finite-shot noisy "
            "statevector trajectories with shot branching"
        )
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
print("\nExpectation values of local method:")
print("[" + ", ".join(f"{x:.{precision}f}" for x in expectation_values) + "]")


# ============================================================
# Gate counts before/after transpilation
# ============================================================
raw_gate_counts = _gate_count_list(folded_circuits)
transpiled_gate_counts = _gate_count_list(exec_circuits)

# ============================================================
# Linear and exponential extrapolation on four gate-count axes
# ============================================================
report_path, axis_report = _write_report(
    "local",
    expectation_values,
    raw_gate_counts,
    transpiled_gate_counts,
    ideal_expectation=None,
)
_print_gate_axis_results(axis_report, precision)
print(f"\nReport saved to: {report_path}")
