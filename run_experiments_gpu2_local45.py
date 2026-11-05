"""GPU2 add-on: 32 4q/5q Local Folding jobs using eight GPUs.

Dispatch Round 1->4, 4q before 5q, each with Random/CZ/CRZ/RZZ.
No round-completion barriers, no job timeouts, no automatic Git actions.
Keeps ALL original experiment_config_2.py and run_experiments_2.py untouched.
Requires the unchanged run_experiments.py, U_config.py, and local ZNE program.
Does not launch if another ZNE batch or worker is active in this project.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import experiment_config_gpu2_local45 as cfg2
import run_experiments as runner

runner.cfg = cfg2

GPU_IDS = tuple(int(gpu) for gpu in cfg2.GPU_IDS)
PRINT_LOCK = threading.Lock()
MONITORED_PROGRAMS = {
    "run_experiments.py",
    "run_experiments_2.py",
    "run_experiments_gpu2_local45.py",
    "ZNE_circuit_folding.py",
    "ZNE_depth_reduced_global.py",
    "ZNE_depth_reduced_local.py",
}


def say(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Print all jobs and skips; do not run simulations.")
    group.add_argument("--run", action="store_true", help="Run GPU jobs without barriers or timeouts.")
    parser.add_argument("--show-commands", action="store_true", help="In dry-run, print each child command.")
    parser.add_argument("--progress-seconds", type=float, default=cfg2.PROGRESS_UPDATE_SECONDS)
    return parser.parse_args()


def validate_plan(jobs: list[Any]) -> None:
    if cfg2.JOB_TIMEOUT_SECONDS is not None:
        raise ValueError("JOB_TIMEOUT_SECONDS must remain None. No timeout is allowed.")
    if not cfg2.SKIP_EXISTING:
        raise ValueError("SKIP_EXISTING must be True.")
    if not GPU_IDS or len(set(GPU_IDS)) != len(GPU_IDS):
        raise ValueError("GPU_IDS must be a nonempty sequence of distinct GPU indices.")
    if len(jobs) != 32:
        raise ValueError(f"Expected exactly 32 local-folding jobs; got {len(jobs)}.")

    priority = []
    for job in jobs:
        if job.num_qubits not in (4, 5) or job.method != "local":
            raise ValueError(f"Only 4q/5q Local Folding is allowed: {job.num_qubits}q, {job.method}")
        if job.depth_round not in (0, 1, 2, 3) or job.batch_phase != 0:
            raise ValueError(f"Unexpected round/phase: {job.depth_round}, {job.batch_phase}")
        if job.profile_name != "aer_increasing_pauli" or job.backend != "aer_simulator":
            raise ValueError("Only the custom Aer increasing-Pauli profile is allowed.")
        if job.noise != "increasing_random_pauli" or job.report_tag != "GPU_increasing_pauli":
            raise ValueError("Unexpected noise setting or report tag.")
        factors = (1, 3, 5, 7) if job.num_qubits == 4 else (1, 3, 5)
        if job.fold_factors != factors:
            raise ValueError(f"Unexpected fold factors for {job.num_qubits}q: {job.fold_factors}")
        if (job.shots != 1000 or job.base_seed != 150
                or job.gate_weight_1q != .3 or job.gate_weight_2q != .7
                or job.max_simulator_physical_qubits != 40
                or job.machine.lower() != "gpu"):
            raise ValueError(f"Experiment parameters differ from original Ver2: {job}")
        if Path(job.report_directory).name != "zne_reports_ver2":
            raise ValueError("Report directory must be zne_reports_ver2.")
        priority.append((job.depth_round, job.num_qubits))

    if priority != sorted(priority):
        raise ValueError("Wrong dispatch order: expected Round first, then increasing qubit count.")
    expected_counts = {(r, q): 4 for r in range(4) for q in (4, 5)}
    if Counter(priority) != expected_counts:
        raise ValueError("Wrong number of jobs per (round, qubit) group.")


def valid_matching_report(job: Any) -> bool:
    """Validate old Ver2 local reports before SKIP. Never modify report files."""
    if not runner._valid_existing_report(job):
        return False
    try:
        report = json.loads(job.report_path.read_text(encoding="utf-8"))
        execution = report["execution"]
        noise = report["noise"]
        depth_noise = noise["depth_dependent_noise_configuration"]
        parameters = depth_noise["initial_parameters"]
        weights = report["weights"]
        values = report["expectation_values"]
        if len(values) != len(job.fold_factors):
            return False
        if any(not isinstance(v, (int, float)) for v in values):
            return False
        if execution.get("backend_kind") != job.backend:
            return False
        if execution.get("simulation_device", "").lower() != "gpu":
            return False
        if execution.get("shots") != job.shots:
            return False
        if execution.get("noise_filename_tag") != "increasing_pauli":
            return False
        if noise.get("selected_custom_noise_family") != "depth_dependent":
            return False
        if noise.get("active_depth_dependent_noise_types") != ["depth_dependent_random_pauli"]:
            return False
        if depth_noise.get("base_seed") != job.base_seed:
            return False
        if (float(parameters["p_1q"]) != .000245
                or float(parameters["p_2q"]) != .006970):
            return False
        if (float(depth_noise.get("reference_depth")) != 100.0
                or float(depth_noise.get("noise_multiplier")) != 3.0
                or float(depth_noise.get("maximum_noise_multiplier")) != 4.0):
            return False
        if (float(weights["w1_one_qubit"]) != job.gate_weight_1q
                or float(weights["w2_two_qubit"]) != job.gate_weight_2q):
            return False
        return True
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError):
        return False


def existing_or_pending(jobs: list[Any]) -> tuple[list[Any], list[Any]]:
    """Never replace an existing report, including one with incompatible content."""
    completed, pending = [], []
    conflicts = []
    for job in jobs:
        if job.report_path.exists():
            if valid_matching_report(job):
                completed.append(job)
            else:
                conflicts.append(str(job.report_path))
        else:
            pending.append(job)
    if conflicts:
        raise RuntimeError(
            "STOP: Existing report(s) do not match the requested settings or are invalid. "
            "They have NOT been modified. Inspect/move them manually before running:\n  "
            + "\n  ".join(conflicts[:30])
            + (f"\n  ... and {len(conflicts) - 30} more" if len(conflicts) > 30 else "")
        )
    return completed, pending


def print_plan(jobs, completed, pending, *, show_commands):
    say(f"Project root      : {Path(__file__).resolve().parent}")
    say(f"Configured jobs   : {len(jobs)}")
    say(f"Valid skips       : {len(completed)}")
    say(f"Pending jobs      : {len(pending)}")
    say(f"Physical GPUs     : {', '.join(map(str, GPU_IDS))}")
    say(f"Report directory  : {cfg2.REPORT_DIRECTORY}")
    say(f"Log directory     : {cfg2.LOG_DIRECTORY}")
    say("All existing reports: READ-ONLY; matching reports are skipped")
    say("Noise             : increasing_random_pauli ONLY")
    say("Scheduling        : Round -> 4q/5q -> circuit -> Local (DISPATCH priority)")
    say("Completion barrier: NONE; idle GPUs take next pending job immediately")
    say("Job timeout       : NONE (permanently disabled)")
    last_key = None
    for job in jobs:
        key = (job.depth_round + 1, job.num_qubits)
        if key != last_key:
            say(f"\n=== ROUND {key[0]} | {key[1]}q ===")
            last_key = key
        status = "SKIP" if job.report_path.exists() else "PLAN"
        say(f"[{job.index:04d}/{job.total:04d}] {status} | {runner._job_description(job)} | factors={list(job.fold_factors)}")
        if show_commands:
            say("    " + " ".join(runner._command(job)))
    say("\nDry run complete. No simulations or report files were modified.")


def active_project_jobs(project_root: Path) -> list[str]:
    """Guard against starting a second 8-GPU batch on top of a running one."""
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("GPU batch launch is supported only on Linux with /proc.")
    running = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = [a.decode("utf-8", "replace") for a in (entry / "cmdline").read_bytes().split(b"\0") if a]
            if not argv or not any(Path(a).name in MONITORED_PROGRAMS for a in argv):
                continue
            cwd = (entry / "cwd").resolve()
            in_project = cwd == project_root or any(
                Path(a).is_absolute() and Path(a).resolve().parent == project_root
                for a in argv if a.endswith(".py")
            )
            if in_project:
                running.append(f"PID {entry.name}: {' '.join(argv)[:180]}")
        except (OSError, PermissionError, ProcessLookupError):
            continue
    return running


def acquire_run_lock(project_root: Path):
    """Stop accidental concurrent launches of this *new* runner."""
    lock_path = project_root / ".gpu2_local45_runner.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"STOP: Another ver2 runner already holds {lock_path}.")
    return handle


def preflight_gpu(project_root: Path) -> None:
    code = (
        "from qiskit_aer import AerSimulator; "
        "d=tuple(AerSimulator().available_devices()); "
        "assert 'GPU' in d, f'Aer GPU unavailable: {d}'; "
        "print(d)"
    )
    for gpu_id in GPU_IDS:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        checked = subprocess.run(
            [sys.executable, "-c", code], cwd=project_root, env=env,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if checked.returncode:
            raise RuntimeError(f"GPU {gpu_id} preflight failed:\n{checked.stdout.strip()}")
        say(f"GPU {gpu_id} preflight: PASS")


def run_job(job: Any, gpu_id: int, project_root: Path, log_root: Path,
            batch_start: float, progress_seconds: float) -> dict[str, Any]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path = log_root / f"gpu{gpu_id}_{job.slug}.log"
    start = time.monotonic()

    def heartbeat(elapsed: float):
        say(
            f"[{job.index:04d}/{job.total:04d}] RUNNING | GPU={gpu_id} | "
            f"job={runner._format_duration(elapsed)} | "
            f"batch={runner._format_duration(time.monotonic() - batch_start)} | "
            f"{runner._job_description(job)}"
        )

    # Absolutely no elapsed-time timeout. We do not call kill/terminate here.
    rc = runner._run_child(
        runner._command(job), cwd=project_root, env=env, log_path=log_path,
        stream_output=cfg2.STREAM_OUTPUT, progress_seconds=progress_seconds,
        timeout_seconds=None, heartbeat=heartbeat,
    )
    elapsed = time.monotonic() - start
    valid = rc == 0 and valid_matching_report(job)
    return {
        "returncode": rc, "runtime": elapsed, "report_valid": valid,
        "log_path": str(log_path),
    }


def save_summary(path: Path, results: list[dict[str, Any]]) -> None:
    # Rewriting the log summary does not touch any report or simulation process.
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    temp.replace(path)


def run_batch(project_root: Path, jobs: list[Any], completed: list[Any],
              pending: list[Any], *, progress_seconds: float) -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = (project_root / cfg2.LOG_DIRECTORY / timestamp).resolve()
    log_root.mkdir(parents=True, exist_ok=False)
    summary_path = log_root / "batch_summary.json"
    results: list[dict[str, Any]] = [
        runner._job_record(job, status="skipped", gpu_id=None, runtime_seconds=0.0,
                           report_path=str(job.report_path))
        for job in completed
    ]
    save_summary(summary_path, results)

    successes, failures, skipped = 0, 0, len(completed)
    available = deque(GPU_IDS)
    remaining = deque(pending)
    active = {}
    start_time = time.monotonic()
    say(f"Total={len(jobs)}; skipped={skipped}; pending={len(remaining)}")

    # ONE global queue and ONE thread pool for ALL rounds and ALL qubit counts.
    # No barrier joins or waits for a qubit group/round to finish.
    with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
        while remaining or active:
            while remaining and available:
                job = remaining.popleft()
                # An external sync might have placed a valid report after the
                # initial audit. Never overwrite or redundantly run that job.
                if job.report_path.exists():
                    if not valid_matching_report(job):
                        raise RuntimeError(f"STOP: existing incompatible report: {job.report_path}")
                    skipped += 1
                    results.append(runner._job_record(
                        job, status="skipped", gpu_id=None,
                        runtime_seconds=0.0, report_path=str(job.report_path)))
                    save_summary(summary_path, results)
                    say(f"[{job.index:04d}/{job.total:04d}] SKIP (appeared later) | {job.report_filename}")
                    continue
                gpu_id = available.popleft()
                say(f"[{job.index:04d}/{job.total:04d}] START | GPU={gpu_id} | "
                    f"round={job.depth_round + 1} | {runner._job_description(job)}")
                future = executor.submit(
                    run_job, job, gpu_id, project_root, log_root,
                    start_time, progress_seconds,
                )
                active[future] = (job, gpu_id)

            if not active:
                continue

            done, _ = wait(set(active), return_when=FIRST_COMPLETED)
            for future in done:
                job, gpu_id = active.pop(future)
                available.append(gpu_id)
                try:
                    outcome = future.result()
                except Exception as exc:
                    failures += 1
                    say(f"[{job.index:04d}/{job.total:04d}] FAILED | GPU={gpu_id} | runner error: {exc!r}")
                    results.append(runner._job_record(
                        job, status="failed", gpu_id=gpu_id,
                        runner_exception=repr(exc), report_path=str(job.report_path)))
                    save_summary(summary_path, results)
                    continue

                status = "success" if outcome["report_valid"] else "failed"
                if status == "success":
                    successes += 1
                else:
                    failures += 1
                say(f"[{job.index:04d}/{job.total:04d}] {status.upper()} | GPU={gpu_id} | "
                    f"time={runner._format_duration(outcome['runtime'])} | "
                    f"round={job.depth_round + 1} | {runner._job_description(job)}")
                results.append(runner._job_record(
                    job, status=status, gpu_id=gpu_id,
                    returncode=outcome["returncode"],
                    runtime_seconds=outcome["runtime"],
                    report_path=str(job.report_path), log_path=outcome["log_path"],
                ))
                save_summary(summary_path, results)

    say("\n================== GPU2 LOCAL45 BATCH COMPLETE ==================")
    say(f"Processed: {successes + failures + skipped}/{len(jobs)}")
    say(f"Succeeded: {successes} | Failed: {failures} | Skipped: {skipped}")
    say(f"Elapsed: {runner._format_duration(time.monotonic() - start_time)}")
    say(f"Summary: {summary_path}")
    say(f"Job logs: {log_root}")
    return 1 if failures else 0


def main() -> int:
    args = arguments()
    project_root = Path(__file__).resolve().parent
    runner.cfg = cfg2
    if args.progress_seconds < 0:
        raise ValueError("--progress-seconds must be nonnegative")

    jobs = runner._build_jobs(project_root, profile_filter=None,
                              experiment_filter=None, method_filter=None)
    validate_plan(jobs)
    warnings = runner._validate_jobs(jobs, runner._load_u_config(project_root))
    for warning in dict.fromkeys(warnings):
        say("WARNING: " + warning)

    completed, pending = existing_or_pending(jobs)
    if args.dry_run or not args.run:
        print_plan(jobs, completed, pending, show_commands=args.show_commands)
        return 0

    # Fail safely if the user's original GPU2 8-GPU batch is still running.
    if not (project_root / cfg2.REPORT_DIRECTORY).is_dir():
        raise RuntimeError(
            "STOP: zne_reports_ver2 does not exist on this GPU. Transfer the "
            "previous report files BEFORE running so they can be skipped."
        )
    if not list((project_root / cfg2.REPORT_DIRECTORY).glob("*.json")):
        raise RuntimeError(
            "STOP: zne_reports_ver2 is empty on this GPU. Check that previously "
            "completed reports were copied from the Mac."
        )

    lock = acquire_run_lock(project_root)
    try:
        other_jobs = active_project_jobs(project_root)
        if other_jobs:
            raise RuntimeError(
                "STOP: another ZNE batch or worker is still running in this "
                "project. No new GPU experiment was started:\n  "
                + "\n  ".join(other_jobs[:20])
            )
        say("Preflight: checking all requested GPUs...")
        preflight_gpu(project_root)
        # Re-read matching reports immediately before dispatching jobs.
        completed, pending = existing_or_pending(jobs)
        return run_batch(project_root, jobs, completed, pending,
                         progress_seconds=args.progress_seconds)
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
