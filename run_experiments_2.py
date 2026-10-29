"""Run GPU-2 Round 3/4 on eight physical GPUs with strict barriers.

The repository's run_experiments.py remains the source of truth for job
expansion, validation, report naming, report validation, and child commands.
This wrapper adds only multi-GPU scheduling.

Scheduling rule
---------------
Jobs are grouped by:
    phase -> depth round -> qubit count

A group is a strict qubit barrier. Up to eight jobs inside that qubit group run
in parallel, one child process per physical GPU. Jobs remain queued in the base
order (random_type -> hea_cz -> hea_crz -> hea_rzz, then profile/method order).
The next qubit count never starts until every job for the current qubit count
has finished. Therefore the original Round/qubit ordering cannot be overtaken
by a faster GPU, while all eight GPUs can stay busy across circuit families.

There is NO elapsed-time timeout.  Every child calls the base runner with
``timeout_seconds=None`` regardless of any external setting.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import experiment_config_2 as cfg2
import run_experiments as runner


# ---------------------------------------------------------------------------
# Make the existing base runner use the GPU-2 configuration.
# ---------------------------------------------------------------------------
runner.cfg = cfg2

_base_iter_experiment_points = runner._iter_experiment_points


def _iter_experiment_points_with_offset(experiment):
    offset = int(experiment.get("depth_round_offset", 0))
    for num_qubits, depth, depth_round in _base_iter_experiment_points(experiment):
        yield num_qubits, depth, depth_round + offset


runner._iter_experiment_points = _iter_experiment_points_with_offset

# Permanent safety contract for this project.
if getattr(cfg2, "JOB_TIMEOUT_SECONDS", None) is not None:
    raise RuntimeError(
        "experiment_config_2.py must keep JOB_TIMEOUT_SECONDS = None."
    )

GPU_IDS = tuple(int(value) for value in getattr(cfg2, "GPU_IDS", ()))
if not GPU_IDS:
    raise RuntimeError("experiment_config_2.py must define at least one GPU in GPU_IDS.")
if len(set(GPU_IDS)) != len(GPU_IDS):
    raise RuntimeError(f"GPU_IDS must be unique; got {GPU_IDS!r}.")

_PRINT_LOCK = threading.Lock()


def _safe_print(*args, **kwargs) -> None:
    with _PRINT_LOCK:
        print(*args, **kwargs, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only.")
    mode.add_argument("--run", action="store_true", help="Execute simulations.")
    parser.add_argument("--profile", action="append", help="Run only named profile(s).")
    parser.add_argument("--experiment", action="append", help="Run only named experiment group(s).")
    parser.add_argument("--method", action="append", choices=tuple(cfg2.METHOD_FILES))
    parser.add_argument("--limit", type=int, help="Limit expanded jobs after filtering.")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--no-git", action="store_true")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Also print all child output. Not recommended with eight concurrent jobs.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=float,
        help="Override concise heartbeat interval.",
    )
    parser.add_argument(
        "--show-commands",
        action="store_true",
        help="In dry-run mode, show child commands.",
    )
    return parser.parse_args()


def _circuit_order() -> dict[str, int]:
    return {
        str(experiment["name"]): index
        for index, experiment in enumerate(cfg2.EXPERIMENTS)
        if experiment.get("enabled", True)
    }


def _barrier_key(job) -> tuple[int, int, int]:
    return (
        int(job.batch_phase),
        int(job.depth_round),
        int(job.num_qubits),
    )


def _group_by_barrier(jobs) -> list[tuple[tuple[int, int, int], list[Any]]]:
    groups: list[tuple[tuple[int, int, int], list[Any]]] = []
    for job in jobs:
        key = _barrier_key(job)
        if not groups or groups[-1][0] != key:
            groups.append((key, [job]))
        else:
            groups[-1][1].append(job)
    return groups


def _barrier_label(key, jobs) -> str:
    phase, depth_round, num_qubits = key
    circuits = []
    for job in jobs:
        if job.unitary not in circuits:
            circuits.append(job.unitary)
    return (
        f"phase={phase} | Round {depth_round + 1} | {num_qubits}q | "
        f"circuits={'/'.join(circuits)} | jobs={len(jobs)}"
    )


def _validate_plan(jobs) -> None:
    if getattr(cfg2, "JOB_TIMEOUT_SECONDS", None) is not None:
        raise RuntimeError("Timeout must remain disabled (JOB_TIMEOUT_SECONDS = None).")

    rounds = {int(job.depth_round) + 1 for job in jobs}
    if not rounds.issubset({3, 4}):
        raise RuntimeError(f"GPU2 plan must contain only Round 3/4; got {sorted(rounds)}")

    # Reuse the base runner's strict phase/round/qubit validation if present.
    validator = getattr(runner, "_validate_strict_batch_order", None)
    if validator is not None:
        validator(jobs)

    # Extra validation: barriers themselves may never regress.
    keys = [_barrier_key(job) for job in jobs]
    if keys != sorted(keys):
        raise RuntimeError("Barrier order regression detected in expanded GPU2 jobs.")


def _preflight_each_gpu(project_root: Path) -> None:
    """Verify Qiskit Aer sees a GPU when each physical device is isolated."""
    code = (
        "from qiskit_aer import AerSimulator; "
        "d=tuple(AerSimulator().available_devices()); "
        "assert 'GPU' in d, f'Aer GPU unavailable: {d}'; "
        "print(d)"
    )
    for gpu_id in GPU_IDS:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"GPU {gpu_id} preflight failed:\n{completed.stdout.strip()}"
            )
        _safe_print(f"Preflight GPU {gpu_id}: PASS")


def _run_one_job(
    job,
    gpu_id: int,
    *,
    project_root: Path,
    log_root: Path,
    stream_output: bool,
    progress_seconds: float,
    batch_start: float,
) -> dict[str, Any]:
    env = dict(os.environ)
    # Critical: each child sees exactly one physical GPU.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("PYTHONUNBUFFERED", "1")

    command = runner._command(job)
    log_path = log_root / f"gpu{gpu_id}_{job.slug}.log"
    start = time.monotonic()

    def heartbeat(job_elapsed: float) -> None:
        _safe_print(
            f"[{job.index:04d}/{job.total:04d}] RUNNING | GPU={gpu_id} | "
            f"job={runner._format_duration(job_elapsed)} | "
            f"batch={runner._format_duration(time.monotonic() - batch_start)} | "
            f"{runner._job_description(job)}"
        )

    # Hard-coded None: this multi-GPU runner never kills a job due to runtime.
    returncode = runner._run_child(
        command,
        cwd=project_root,
        env=env,
        log_path=log_path,
        stream_output=stream_output,
        progress_seconds=progress_seconds,
        timeout_seconds=None,
        heartbeat=heartbeat,
    )
    runtime = time.monotonic() - start
    report_valid = returncode == 0 and runner._valid_existing_report(job)

    return {
        "job": job,
        "gpu_id": gpu_id,
        "returncode": returncode,
        "runtime": runtime,
        "report_valid": report_valid,
        "log_path": log_path,
    }


def _write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parent
    u_config = runner._load_u_config(project_root)

    known_profiles = set(cfg2.EXECUTION_PROFILES)
    known_experiments = {str(item["name"]) for item in cfg2.EXPERIMENTS}
    if args.profile:
        unknown = set(args.profile) - known_profiles
        if unknown:
            raise ValueError(f"Unknown --profile value(s): {sorted(unknown)}")
    if args.experiment:
        unknown = set(args.experiment) - known_experiments
        if unknown:
            raise ValueError(f"Unknown --experiment value(s): {sorted(unknown)}")

    jobs = runner._build_jobs(
        project_root,
        profile_filter=set(args.profile) if args.profile else None,
        experiment_filter=set(args.experiment) if args.experiment else None,
        method_filter=set(args.method) if args.method else None,
    )
    if not jobs:
        raise ValueError("The selected filters generated zero jobs.")

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        jobs = jobs[: args.limit]
        jobs = [
            runner.Job(**{**asdict(job), "index": i + 1, "total": len(jobs)})
            for i, job in enumerate(jobs)
        ]

    warnings = runner._validate_jobs(jobs, u_config)
    _validate_plan(jobs)

    dry_run = cfg2.DRY_RUN
    if args.dry_run:
        dry_run = True
    elif args.run:
        dry_run = False

    stream_output = bool(cfg2.STREAM_OUTPUT or args.stream_output)
    progress_seconds = (
        float(args.progress_seconds)
        if args.progress_seconds is not None
        else float(cfg2.PROGRESS_UPDATE_SECONDS)
    )
    if progress_seconds < 0:
        raise ValueError("--progress-seconds must be zero or positive.")

    print("=" * 96)
    print("ZNE GPU2 MULTI-GPU BATCH PLAN")
    print(f"Project root     : {project_root}")
    print(f"Jobs             : {len(jobs)}")
    print(f"Mode             : {'DRY RUN' if dry_run else 'EXECUTE'}")
    print(f"Physical GPUs    : {', '.join(map(str, GPU_IDS))}")
    print(f"Parallel workers : {len(GPU_IDS)}")
    print("Job timeout      : disabled permanently")
    print(f"Reports          : {cfg2.REPORT_DIRECTORY}")
    print(f"Logs             : {cfg2.LOG_DIRECTORY}")
    print(
        "Barrier order    : phase -> Round -> qubit; "
        "circuits/profile/method jobs parallel inside each qubit barrier"
    )
    print("=" * 96)

    for warning in dict.fromkeys(warnings):
        print("WARNING:", warning)
    if warnings:
        print()

    barriers = _group_by_barrier(jobs)

    if dry_run:
        for key, members in barriers:
            print("\n--- BARRIER | " + _barrier_label(key, members) + " ---")
            for position, job in enumerate(members):
                gpu_id = GPU_IDS[position % len(GPU_IDS)]
                print(
                    f"[{job.index:04d}/{job.total:04d}] PLAN | GPU~{gpu_id} | "
                    f"round={job.depth_round + 1} | {runner._job_description(job)}"
                )
                if args.show_commands:
                    print("    command:", " ".join(runner._command(job)))
        print("\nDry run complete. No simulations were executed.")
        return 0

    if stream_output and len(GPU_IDS) > 1:
        print(
            "WARNING: --stream-output with multiple GPUs will interleave child output; "
            "per-job log files remain authoritative."
        )

    _safe_print("Preflight: checking each configured GPU ...")
    _preflight_each_gpu(project_root)
    _safe_print("Preflight: all GPUs PASS\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = (project_root / cfg2.LOG_DIRECTORY / timestamp).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    summary_path = log_root / "batch_summary.json"

    if args.no_git:
        cfg2.GIT["enabled"] = False

    results: list[dict[str, Any]] = []
    successes = 0
    failures = 0
    skipped = 0
    batch_start = time.monotonic()
    stop_after_barrier = False

    for barrier_index, (key, members) in enumerate(barriers, start=1):
        _safe_print(
            "\n" + "=" * 96 +
            f"\nBARRIER {barrier_index}/{len(barriers)} | {_barrier_label(key, members)}" +
            "\n" + "=" * 96
        )

        pending = deque()
        for job in members:
            job.report_directory.mkdir(parents=True, exist_ok=True)
            if (
                cfg2.SKIP_EXISTING
                and not args.rerun_existing
                and runner._valid_existing_report(job)
            ):
                skipped += 1
                _safe_print(
                    f"[{job.index:04d}/{job.total:04d}] SKIP | "
                    f"{runner._job_description(job)}"
                )
                results.append(
                    runner._job_record(
                        job,
                        status="skipped",
                        gpu_id=None,
                        runtime_seconds=0.0,
                        report_path=str(job.report_path),
                    )
                )
            else:
                pending.append(job)

        _write_summary(summary_path, results)

        if not pending:
            _safe_print("Barrier already complete from existing valid reports.")
            continue

        available_gpus = deque(GPU_IDS)
        active: dict[Future, tuple[Any, int]] = {}
        scheduling_open = True

        with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
            while pending or active:
                while scheduling_open and pending and available_gpus:
                    job = pending.popleft()
                    gpu_id = available_gpus.popleft()
                    _safe_print(
                        f"[{job.index:04d}/{job.total:04d}] START | GPU={gpu_id} | "
                        f"{runner._job_description(job)}"
                    )
                    future = executor.submit(
                        _run_one_job,
                        job,
                        gpu_id,
                        project_root=project_root,
                        log_root=log_root,
                        stream_output=stream_output,
                        progress_seconds=progress_seconds,
                        batch_start=batch_start,
                    )
                    active[future] = (job, gpu_id)

                if not active:
                    break

                done, _ = wait(set(active), return_when=FIRST_COMPLETED)
                for future in done:
                    job, gpu_id = active.pop(future)
                    available_gpus.append(gpu_id)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        failures += 1
                        _safe_print(
                            f"[{job.index:04d}/{job.total:04d}] FAILED | GPU={gpu_id} | "
                            f"runner exception: {exc!r}"
                        )
                        results.append(
                            runner._job_record(
                                job,
                                status="failed",
                                gpu_id=gpu_id,
                                runner_exception=repr(exc),
                                report_path=str(job.report_path),
                            )
                        )
                        if cfg2.STOP_ON_FAILURE:
                            scheduling_open = False
                            stop_after_barrier = True
                        _write_summary(summary_path, results)
                        continue

                    runtime = float(outcome["runtime"])
                    returncode = int(outcome["returncode"])
                    report_valid = bool(outcome["report_valid"])
                    log_path = Path(outcome["log_path"])

                    if report_valid:
                        successes += 1
                        _safe_print(
                            f"[{job.index:04d}/{job.total:04d}] SUCCESS | GPU={gpu_id} | "
                            f"job={runner._format_duration(runtime)} | "
                            f"{runner._job_description(job)}"
                        )
                        status = "success"
                    else:
                        failures += 1
                        # No timeout path exists in this runner.
                        _safe_print(
                            f"[{job.index:04d}/{job.total:04d}] FAILED | GPU={gpu_id} | "
                            f"rc={returncode} | job={runner._format_duration(runtime)} | "
                            f"{runner._job_description(job)}"
                        )
                        _safe_print(f"    log: {log_path}")
                        status = "failed"
                        if cfg2.STOP_ON_FAILURE:
                            scheduling_open = False
                            stop_after_barrier = True

                    results.append(
                        runner._job_record(
                            job,
                            status=status,
                            gpu_id=gpu_id,
                            returncode=returncode,
                            runtime_seconds=runtime,
                            report_path=str(job.report_path),
                            log_path=str(log_path),
                        )
                    )
                    _write_summary(summary_path, results)

        if pending and not scheduling_open:
            _safe_print(
                f"STOP_ON_FAILURE=True: {len(pending)} job(s) in this barrier were not started."
            )

        # Strict qubit barrier: the next qubit count starts only here, after every
        # active job in this group has returned.
        _safe_print(
            f"BARRIER COMPLETE | {_barrier_label(key, members)} | "
            f"S={successes} F={failures} K={skipped}"
        )

        if stop_after_barrier:
            break

    if cfg2.GIT.get("enabled", False):
        runner._git_checkpoint(project_root, f"Add GPU2 multi-GPU results {timestamp}")

    elapsed = time.monotonic() - batch_start
    processed = successes + failures + skipped
    print("\n" + "=" * 96)
    print("GPU2 MULTI-GPU BATCH COMPLETE")
    print(f"Processed        : {processed}/{len(jobs)}")
    print(f"Succeeded        : {successes}")
    print(f"Skipped          : {skipped}")
    print(f"Failed           : {failures}")
    print(f"Elapsed          : {runner._format_duration(elapsed)}")
    print(f"Summary          : {summary_path}")
    print(f"Per-job logs     : {log_root}")
    print("=" * 96)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
