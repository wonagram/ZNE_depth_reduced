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
    "profiles": [
        "aer_small_constant_depolarizing",
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

    # Phase 0: custom-Aer profiles run first in depth-round order.
    "aer_small_constant_depolarizing": {
        "backend": "aer_simulator",
        "noise": "small_constant_depolarizing",
        "report_tag": "GPU_small_constant_depolarizing_0p2pct",
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
#   A) Complete EVERY normal-Aer backend/circuit/qubit/method job in
#      depth round 1 before ANY job in depth round 2; then round 3, round 4, etc.
#      Thus a normal-Aer round-1 depth (roughly 24/27/30/32) never appears after
#      a normal-Aer round-2/3/4 job has started.
#   B) Only after all normal Aer depth rounds finish, run the complete
#      FakeQuebec statevector-trajectory phase, also in depth-major order.
# random_type joins each round by ordinal position because its depths differ.
# ============================================================
EXPERIMENTS = [

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
        "name": "hea_rzz_complete_repeats",
        "unitary": "hea_rzz",
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
            # 30 is also a complete boundary (5 repeats).  Use the same
            # 30-step sweep as the other circuit families.
            2: [30, 60, 90, 120],
            3: [30, 60, 90, 120],
            4: [30, 60, 90, 120],
            5: [30, 60, 90, 120],
        },
    },

    # random_type preserves the existing 1-4q behavior.  The new 5q circuit
    # uses the requested deterministic 8-depth RX/RY/CX/CX/RY/RZ/CX/CX repeat.
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
            # 5q requested RX/RY/CX/CX/RY/RZ/CX/CX schedule: period 8.
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
