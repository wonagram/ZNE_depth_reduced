"""Batch configuration for the ZNE GPU experiment runner.

Normally this is the only file you edit before launching a batch.
"""

from __future__ import annotations

# ============================================================
# Safety and execution behavior
# ============================================================
# True: print/validate every generated job, but run no simulation.
# False: execute the jobs.
DRY_RUN = True

# A valid existing JSON report is skipped. This makes interrupted batches resumable.
SKIP_EXISTING = True

# False (recommended for a large batch): the main-file output is written only
# to each job's log file. The terminal shows concise progress/status messages.
# True: stream the full main-file output to the terminal as well.
# Running a main file directly is unaffected and still prints its normal output.
STREAM_OUTPUT = False

# While a quiet child job is running, print one heartbeat at this interval.
# Set to 0 to disable heartbeat messages.
PROGRESS_UPDATE_SECONDS = 30

# Continue to the next job after a failure.
STOP_ON_FAILURE = False

# Prevent one unexpectedly expensive simulation from blocking every later job.
# A timed-out child process is terminated, recorded as FAILED/TIMEOUT, and the
# batch continues. Set to 0 or None to disable the timeout.
JOB_TIMEOUT_SECONDS = 12 * 60 * 60  # 12 hours per job

# Select the visible NVIDIA GPU(s). "0" means the first GPU.
CUDA_VISIBLE_DEVICES = "0"

# ============================================================
# Project paths
# ============================================================
REPORT_DIRECTORY = "./zne_reports"
LOG_DIRECTORY = "./batch_logs"

METHOD_FILES = {
    "circuit_folding": "ZNE_circuit_folding.py",
    "global": "ZNE_depth_reduced_global.py",
    "local": "ZNE_depth_reduced_local.py",
}

# ============================================================
# Settings shared by all experiments unless overridden in a block
# ============================================================
COMMON = {
    "machine": "gpu",
    "methods": "all",
    # The dedicated FakeQuebec smoke profile is excluded here and is used only
    # by the small first experiment below.
    "profiles": [
        "aer_constant_depolarizing",
        "aer_increasing_pauli",
        "aer_increasing_pauli_amplitude_damping",
        "fake_quebec_statevector",
    ],
    "shots": 1000,
    # Default folding factors for 1-4 logical qubits.
    "fold_factors": [1, 3, 5, 7],

    # Per-qubit override understood by run_experiments.py.  The 5q global/local
    # circuits would otherwise use 5*7=35 physical wires.  Restricting 5q to
    # [1, 3, 5] lowers the maximum to 5*5=25 physical wires.
    "fold_factors_by_num_qubits": {
        5: [1, 3, 5],
    },

    "base_seed": 150,
    "gate_weight_1q": 0.3,
    "gate_weight_2q": 0.7,

    # Global/local folding use NUM_U_QUBITS * max(fold_factors) physical
    # wires in the current construction.  With the override above, 5q uses at
    # most 25 wires.  Raising this guard only permits construction; it does not
    # guarantee that GPU memory or runtime is sufficient.
    "max_simulator_physical_qubits": 40,
}

# ============================================================
# Four requested backend/noise profiles
# ============================================================
EXECUTION_PROFILES = {
    # Phase -1: a small three-method smoke test runs before everything else so
    # a backend/API incompatibility is discovered before the overnight batch.
    "fake_quebec_statevector_smoke": {
        "backend": "fake_statevector",
        "fake_backend_name": "FakeQuebec",
        "noise": "none",
        "report_tag": "GPU_FakeQuebec_statevector",
        "allowed_num_qubits": [2],
        "batch_phase": -1,
        "phase_label": "PHASE A - FakeQuebec smoke test",
        "order_priority": 0,
    },

    # Phase 0: custom-Aer profiles run first in depth-round order.
    "aer_constant_depolarizing": {
        "backend": "aer_simulator",
        "noise": "constant_depolarizing",
        "report_tag": "GPU_constant_depolarizing",
        "batch_phase": 0,
        "phase_label": "PHASE B - Normal Aer depth-major experiments",
        "order_priority": 0,
    },
    "aer_increasing_pauli": {
        "backend": "aer_simulator",
        "noise": "increasing_random_pauli",
        "report_tag": "GPU_increasing_pauli",
        "batch_phase": 0,
        "phase_label": "PHASE B - Normal Aer depth-major experiments",
        "order_priority": 1,
    },
    "aer_increasing_pauli_amplitude_damping": {
        "backend": "aer_simulator",
        "noise": "increasing_random_pauli_amplitude_damping",
        "report_tag": "GPU_increasing_pauli_amplitude_damping",
        "batch_phase": 0,
        "phase_label": "PHASE B - Normal Aer depth-major experiments",
        "order_priority": 2,
    },

    # Phase 1 runs last. This is not the ideal statevector backend: FakeQuebec
    # snapshot noise remains active, Aer is forced to statevector trajectories,
    # results use finite shots, and shot branching is enabled.
    "fake_quebec_statevector": {
        "backend": "fake_statevector",
        "fake_backend_name": "FakeQuebec",
        "noise": "none",
        "report_tag": "GPU_FakeQuebec_statevector",
        "allowed_num_qubits": [2, 3, 4, 5],
        "batch_phase": 1,
        "phase_label": "PHASE C - Full FakeQuebec after all normal Aer jobs",
        "order_priority": 0,
    },
}

# ============================================================
# Experiment groups
#
# Each group may use depths_by_num_qubits when different qubit counts require
# different complete-repeat boundaries. Add as many groups as needed.
#
# STRICT ORDER CONTRACT:
#   A) Run the three-job FakeQuebec smoke test first. Existing smoke reports are
#      skipped one exact case at a time, just like every other job.
#   B) For normal Aer, the order is STRICTLY:
#         round -> qubit count -> circuit family -> noise profile -> ZNE method
#      with qubits 1q -> 2q -> 3q -> 4q -> 5q and circuit families
#      random_type -> hea_cz -> hea_crz -> hea_rzz whenever they exist.
#      Therefore ALL round-1 work for 1q is considered before 2q, ALL 2q work
#      before 3q, etc. Existing reports only skip that exact job; they never
#      cause the runner to resume from a later qubit/circuit block.
#   C) Only after all normal-Aer rounds finish, run the complete FakeQuebec
#      statevector-trajectory phase, using the same round/qubit/circuit order.
# random_type joins each round by ordinal position because its depths differ.
# ============================================================

# Explicit circuit-family order used by run_experiments.py. Do not infer this
# from dictionary/list position: keeping it explicit prevents a later edit from
# accidentally moving hea_rzz ahead of hea_crz, or a heavy 4q case ahead of a
# missing 2q case.
CIRCUIT_ORDER = [
    "random_type",
    "hea_cz",
    "hea_crz",
    "hea_rzz",
]

EXPERIMENTS = [
    # This tests circuit folding, global, and local dynamic execution with the
    # FakeQuebec statevector-trajectory backend before the overnight jobs.
    {
        "name": "fake_quebec_statevector_smoke_test",
        "unitary": "hea_cz",
        "profiles": ["fake_quebec_statevector_smoke"],
        "methods": "all",
        "depths_by_num_qubits": {
            2: [30],
        },
    },

    # Round order for random_type:
    #   R1: 1q=30, 2q=24, 3q=27, 4q=32, 5q=32
    #   R2: 1q=60, 2q=48, 3q=54, 4q=64, 5q=64
    #   R3: 1q=90, 2q=72, 3q=81, 4q=96, 5q=96
    #   R4: 1q=120, 2q=96, 3q=108, 4q=128, 5q=128
    #   R5: only 2q=120 exists.
    {
        "name": "random_type_complete_repeats",
        "unitary": "random_type",
        "depths_by_num_qubits": {
            # 1q repeat: RX -> RY -> RZ (period 3).
            1: [30, 60, 90, 120],
            # 2q full axis/CX-direction phase repeat: period 12.
            2: [24, 48, 72, 96, 120],
            # 3q fixed structural repeat: period 9.
            3: [27, 54, 81, 108],
            # 4q regularized X-Y-X-Z/CX schedule: period 8.
            4: [32, 64, 96, 128],
            # 5q RX/RY/CX/CX/RY/RZ/CX/CX schedule: period 8.
            5: [32, 64, 96, 128],
        },
    },

    {
        "name": "hea_cz_complete_repeats",
        "unitary": "hea_cz",
        "depths_by_num_qubits": {
            2: [30, 60, 90, 120],
            3: [32, 64, 96, 128],
            4: [32, 64, 96, 128],
            5: [32, 64, 96, 128],
        },
    },

    {
        "name": "hea_crz_complete_repeats",
        "unitary": "hea_crz",
        "depths_by_num_qubits": {
            # One 2q HEA-CRZ repeat consumes 6 physical-depth slots, and
            # 30 is also a complete boundary (5 repeats).
            2: [30, 60, 90, 120],
            3: [30, 60, 90, 120],
            4: [30, 60, 90, 120],
            5: [30, 60, 90, 120],
        },
    },

    {
        "name": "hea_rzz_complete_repeats",
        "unitary": "hea_rzz",
        "depths_by_num_qubits": {
            2: [30, 60, 90, 120],
            3: [32, 64, 96, 128],
            4: [32, 64, 96, 128],
            5: [32, 64, 96, 128],
        },
    },
]

# ============================================================
# Optional Git checkpointing
# ============================================================
# Disabled by default. Enable only after confirming that this directory is a
# Git repository and that authentication for `git push` works on the GPU host.
GIT = {
    "enabled": False,
    "push": False,
    "commit_every_successes": 10,
    "remote": "origin",
    "branch": None,  # None means the currently checked-out branch.
    "paths": [REPORT_DIRECTORY, LOG_DIRECTORY],
}
