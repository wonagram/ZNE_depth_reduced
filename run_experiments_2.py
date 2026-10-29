"""Run the GPU-2 Round-3/Round-4 batch using experiment_config_2.py.

This is a thin wrapper around the repository's existing run_experiments.py.
It deliberately reuses the current main runner so ordering, report validation,
resume/skip behavior, and CLI options stay synchronized with GPU-1.
"""

from __future__ import annotations

import experiment_config_2 as cfg2
import run_experiments as runner


# Make the existing runner use the GPU-2 configuration module.
runner.cfg = cfg2

# Preserve the original round numbers.  Because experiment_config_2 keeps only
# two depth entries per circuit, the base runner would otherwise relabel them
# as round 1 and round 2.  The offset restores original Round 3 / Round 4.
_base_iter_experiment_points = runner._iter_experiment_points


def _iter_experiment_points_with_offset(experiment):
    offset = int(experiment.get("depth_round_offset", 0))
    for num_qubits, depth, depth_round in _base_iter_experiment_points(experiment):
        yield num_qubits, depth, depth_round + offset


runner._iter_experiment_points = _iter_experiment_points_with_offset

# Hard safety rule for this GPU-2 runner: no elapsed-time job termination.
if getattr(cfg2, "JOB_TIMEOUT_SECONDS", None) is not None:
    raise RuntimeError(
        "experiment_config_2.py must keep JOB_TIMEOUT_SECONDS = None."
    )


if __name__ == "__main__":
    raise SystemExit(runner.main())
