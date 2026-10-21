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

# Print each child process's output while also saving it to a per-job log file.
STREAM_OUTPUT = True

# Continue to the next job after a failure.
STOP_ON_FAILURE = False

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
    "profiles": "all",
    "shots": 1000,
    "fold_factors": [1, 3, 5, 7],
    "base_seed": 150,
    "gate_weight_1q": 0.3,
    "gate_weight_2q": 0.7,

    # Local folding uses NUM_U_QUBITS * max(fold_factors) physical wires.
    # 5q with fold 7 therefore needs a 35-wire circuit. Raising this guard only
    # permits construction; it does not guarantee that GPU memory is sufficient.
    "max_simulator_physical_qubits": 40,
}

# ============================================================
# Four requested backend/noise profiles
# ============================================================
EXECUTION_PROFILES = {
    "fake_quebec": {
        "backend": "fake_backend",
        "fake_backend_name": "FakeQuebec",
        "noise": "none",
        "report_tag": "GPU_FakeQuebec",
    },
    "aer_constant_depolarizing": {
        "backend": "aer_simulator",
        "noise": "constant_depolarizing",
        "report_tag": "GPU_constant_depolarizing",
    },
    "aer_increasing_pauli": {
        "backend": "aer_simulator",
        "noise": "increasing_random_pauli",
        "report_tag": "GPU_increasing_pauli",
    },
    "aer_increasing_pauli_amplitude_damping": {
        "backend": "aer_simulator",
        "noise": "increasing_random_pauli_amplitude_damping",
        "report_tag": "GPU_increasing_pauli_amplitude_damping",
    },
}

# ============================================================
# Experiment groups
#
# Each group may use depths_by_num_qubits when different qubit counts require
# different complete-repeat boundaries. Add as many groups as needed.
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
            2: [36, 72, 108, 144],
            3: [30, 60, 90, 120],
            4: [30, 60, 90, 120],
            5: [30, 60, 90, 120],
        },
    },

    # random_type keeps the original fixed prefix and extends only the tail
    # with deterministic seeded parameters. It is defined for 1-4 qubits only.
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
