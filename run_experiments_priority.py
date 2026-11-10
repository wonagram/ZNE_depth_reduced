"""Nighthawk r2 + Eagle r3 ZNE 23-stage GPU queue; no completion barriers/timeouts.

Examples:
  python -u run_experiments_priority.py --dry-run
  python -u run_experiments_priority.py --dry-run --show-all
  python -u run_experiments_priority.py --run
  python -u run_experiments_priority.py --status
  python -u run_experiments_priority.py --dry-run --noise-model nighthawk_r2
  python -u run_experiments_priority.py --run --noise-model eagle_r3

Default --noise-model both uses ONLY Nighthawk/Eagle, never Heron.
Eight physical GPUs independently take the next unstarted job in priority order.
Different stages, rounds and hardware-inspired profiles may run concurrently.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import experiment_config_priority as cfg

ROOT = Path(__file__).resolve().parent
GPU_IDS = tuple(int(x) for x in cfg.GPU_IDS)
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class Job:
    index: int
    total: int
    priority: int
    stage: str
    directory_name: str
    noise: str
    noise_model: str
    bell_noise: bool
    round: int
    num_qubits: int
    unitary: str
    depth: int
    method: str
    fold_factors: tuple[int, ...]
    report_tag: str
    legacy_report_name: bool

    @property
    def output_directory(self) -> Path:
        return ROOT / self.directory_name

    @property
    def report_name(self) -> str:
        # No "ver" or other priority label occurs in any report filename.
        return f"{self.num_qubits}q_{self.unitary}_{self.depth}_{self.report_tag}_{self.method}.json"

    @property
    def report_path(self) -> Path:
        return self.output_directory / self.report_name

    @property
    def slug(self) -> str:
        # Version is intentionally present in ONLY the log and status names.
        return (f"{self.index:04d}_P{self.priority}_{self.stage}_r{self.round}_"
                f"{self.num_qubits}q_{self.unitary}_d{self.depth}_{self.method}")

    @property
    def physical_qubits_upper_bound(self) -> int:
        return (self.num_qubits * max(self.fold_factors)
                if self.method in {"global", "local"} else self.num_qubits)


def message(text: str) -> None:
    with PRINT_LOCK:
        print(text, flush=True)


def expand_jobs(noise_model_selection: str = "eagle_r3") -> list[Job]:
    raw = []
    for stage in cfg.selected_stages(noise_model_selection):
        priority = int(stage["priority"])
        mode = str(stage["selection"])
        noise = str(stage["noise"])
        noise_model = str(stage["noise_model"])
        bell = bool(stage["bell_noise"])
        if noise not in cfg.NOISE_FILENAME_TAGS:
            raise ValueError(f"Invalid noise profile {noise!r} in priority {priority}")
        tag = "GPU_" + str(cfg.NOISE_FILENAME_TAGS[noise])
        tag += "_" + {"eagle_r3": "eagle", "nighthawk_r2": "nighthawk"}[noise_model]
        legacy = bool(stage.get('legacy_report_name', False))
        if legacy:
            # Historical ver2 Eagle reports use GPU_increasing_pauli (NO _eagle).
            if (noise, noise_model, bell, mode) != (
                'increasing_random_pauli', 'eagle_r3', False, 'local45'
            ):
                raise AssertionError('Legacy tag unexpectedly used outside Eagle ver2 local45')
            tag = 'GPU_increasing_pauli'
        if bell:
            tag += "_bell_noise"
        for round_number in cfg.ROUND_ORDER:
            for nq in cfg.QUBIT_ORDER:
                methods = cfg.selected_methods(mode, nq)
                if not methods:
                    continue
                factors = tuple(cfg.FOLD_FACTORS_BY_QUBITS[nq])
                for unitary in cfg.CIRCUIT_ORDER:
                    depth = int(cfg.DEPTHS_BY_UNITARY_AND_QUBITS[unitary][nq][round_number - 1])
                    for method in methods:
                        raw.append(dict(
                            priority=priority, stage=str(stage["label"]),
                            directory_name=str(stage["directory"]),
                            noise=noise, noise_model=noise_model,
                            bell_noise=bell, round=round_number,
                            num_qubits=nq, unitary=unitary, depth=depth,
                            method=method, fold_factors=factors, report_tag=tag,
                            legacy_report_name=legacy,
                        ))
    total = len(raw)
    return [Job(index=i + 1, total=total, **entry)
            for i, entry in enumerate(raw)]


def validate_plan(jobs: list[Job]) -> None:
    if not cfg.SKIP_EXISTING:
        raise AssertionError("SKIP_EXISTING must remain True to protect report data")
    if len(set(GPU_IDS)) != len(GPU_IDS) or not GPU_IDS:
        raise AssertionError("GPU_IDS should be distinct nonempty GPU identifiers")
    stages_by_priority = {int(stage["priority"]): stage for stage in cfg.STAGES}
    if tuple(sorted(stages_by_priority)) != tuple(range(1, len(cfg.STAGES) + 1)):
        raise AssertionError("Configured priorities must be unique and consecutive")
    expected_by_selection = {"fast": 160, "local45": 32}
    selected_priorities = {job.priority for job in jobs}
    expected_per_phase = {
        p: expected_by_selection[stage["selection"]]
        for p, stage in stages_by_priority.items() if p in selected_priorities
    }
    if dict(Counter(j.priority for j in jobs)) != expected_per_phase:
        raise AssertionError("Wrong job counts for selected Eagle/Nighthawk stages")
    if len(jobs) != sum(expected_per_phase.values()):
        raise AssertionError("Wrong total job count")
    seen = set()
    previous_priority = None
    for job in jobs:
        sort_key = (job.priority, job.round, job.num_qubits,
                    cfg.CIRCUIT_ORDER.index(job.unitary),
                    cfg.METHOD_ORDER.index(job.method))
        if previous_priority is not None and sort_key < previous_priority:
            raise AssertionError(f"Wrong order: {job.slug}")
        previous_priority = sort_key
        if job.report_path in seen:
            raise AssertionError(f"Two stages would overwrite {job.report_path}")
        seen.add(job.report_path)
        if job.noise not in ("increasing_random_pauli", "increasing_exponential_random_pauli",
                             "constant_seeded_random_pauli"):
            raise AssertionError("Only the three requested Pauli noise profiles are supported")
        stage = stages_by_priority[job.priority]
        if job.noise_model not in cfg.INITIAL_PAULI_PROBABILITIES:
            raise AssertionError(f"Unknown hardware-inspired model: {job.noise_model}")
        if job.noise_model != stage["noise_model"]:
            raise AssertionError(f"Wrong noise model in stage {job.stage}")
        if job.fold_factors != ((1, 3, 5) if job.num_qubits == 5 else (1, 3, 5, 7)):
            raise AssertionError(f"Unexpected folding: {job.slug}")
        if job.physical_qubits_upper_bound > cfg.MAX_SIMULATOR_PHYSICAL_QUBITS:
            raise AssertionError(f"Physical-qubit bound exceeded: {job.slug}")
        if job.method not in cfg.selected_methods(stage["selection"], job.num_qubits):
            raise AssertionError(f"Unexpected method in {job.stage}: {job.method}")
        if "ver" in job.report_name or "bell_noisy" in job.report_name:
            raise AssertionError(f"Invalid report naming: {job.report_name}")
        desired_suffix = {"eagle_r3": "eagle", "nighthawk_r2": "nighthawk"}[job.noise_model]
        expected_tag = ('GPU_increasing_pauli' if job.legacy_report_name else
                        'GPU_' + cfg.NOISE_FILENAME_TAGS[job.noise] + '_' + desired_suffix
                        + ('_bell_noise' if job.bell_noise else ''))
        if job.report_tag != expected_tag:
            raise AssertionError(f"Wrong filename tag: {job.report_name}")
        if job.legacy_report_name != bool(stage.get('legacy_report_name', False)):
            raise AssertionError(f"Legacy naming flag mismatch: {job.slug}")
        if job.legacy_report_name and (job.directory_name != 'zne_reports_ver2' or
               job.method != 'local' or job.num_qubits not in (4, 5) or
               job.noise_model != 'eagle_r3' or job.noise != 'increasing_random_pauli' or
               job.bell_noise):
            raise AssertionError(f"Unsafe legacy job: {job.slug}")
        for other_suffix in ("eagle", "nighthawk", "heron"):
            if other_suffix != desired_suffix and ("_" + other_suffix) in job.report_tag:
                raise AssertionError(f"Wrong hardware label in filename: {job.report_name}")
        if (job.output_directory.name != job.directory_name
                or job.report_tag != expected_tag):
            raise AssertionError("Incorrect report path/tag")


def validate_unitary_runtime_compatibility(jobs: list[Job]) -> None:
    """Fail before launching GPUs when installed U_config / helper cannot run all jobs."""
    import importlib
    import inspect
    try:
        config = importlib.import_module("U_config")
        unitary = importlib.import_module("unitary_tools")
        helper = importlib.import_module("local_folding_helpers")
    except Exception as exc:
        raise RuntimeError("Cannot import U_config, unitary_tools, or local_folding_helpers; zero GPU jobs launched") from exc
    failures = []
    required = sorted({(j.unitary, j.num_qubits, j.depth) for j in jobs})
    for name, n, depth in required:
        try:
            maximum = int(config.get_unitary_max_depth(name, n))
            if maximum < depth:
                raise ValueError(f"maximum supported depth {maximum} is below {depth}")
            selected = config.select_unitary_components(name, n, depth)
            if not selected:
                raise ValueError("empty component selection")
        except Exception as exc:
            failures.append(f"{n}q {name} depth={depth}: {type(exc).__name__}: {exc}")
    for name in ("component_plan", "append_unitary_block", "append_local_component_copy", "configure_aer_simulator_device", "NoiseController"):
        if not hasattr(unitary, name):
            failures.append(f"unitary_tools.py has no {name}")
    for name, params in (("make_local_plan", {"qubit_assignment", "input_placement", "reuse_pattern"}), ("build_intermediate_local_circuit", {"apply_component", "psi_minus", "bell_measure"})):
        func = getattr(helper, name, None)
        if func is None:
            failures.append(f"local_folding_helpers.py has no {name}")
        else:
            missing = params - set(inspect.signature(func).parameters)
            if missing:
                failures.append(f"local_folding_helpers.{name} missing arguments {sorted(missing)}")
    if failures:
        raise RuntimeError("UNITARY PREFLIGHT FAILED; NO GPU JOBS STARTED. Verify the installed U_config.py and helpers.\n" + "\n".join("  " + t for t in failures[:40]) + (f"\n... {len(failures)-40} more" if len(failures)>40 else ""))
    message(f"UNITARY PREFLIGHT PASSED: {len(required)} unique unitary/qubit/depth settings")


def validate_execution_safety(jobs: list[Job]) -> None:
    """Reject accidental drawing/check-code and non-GPU/non-Aer batch commands."""
    import ast
    for filename in cfg.METHOD_FILES.values():
        path = ROOT / filename
        code = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assigns = {target.id: node.value for node in code.body if isinstance(node, ast.Assign) for target in node.targets if isinstance(target, ast.Name)}
        names = ("DRAW_CIRCUIT", "CHECK_CODE") if filename == "ZNE_depth_reduced_local.py" else ("DRAW_CIRCUIT",)
        for name in names:
            val = assigns.get(name)
            if not (isinstance(val, ast.Constant) and val.value is False):
                raise RuntimeError(f"Drawing/check flag {filename}:{name} is not False; no jobs launched")
    for job in jobs:
        cmd = command(job)
        if cmd[cmd.index("--backend")+1] != "aer_simulator" or cmd[cmd.index("--machine")+1] != "gpu":
            raise RuntimeError(f"Unexpected backend/machine: {job.slug}")
        if cmd[cmd.index("--noise-model")+1] not in ("eagle_r3", "nighthawk_r2"):
            raise RuntimeError(f"Unexpected hardware revision: {job.slug}")
    message("SOURCE PREFLIGHT PASSED: drawing OFF, local CHECK_CODE OFF, only GPU Aer Eagle/Nighthawk")


def validate_files_and_methods(jobs: list[Job]) -> None:
    for name in cfg.METHOD_FILES.values():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing ZNE execution file: {path}")
        # Prevent a dangerous mismatch: the runner predicts filenames BEFORE
        # it launches any worker. The same suffix must exist in each main file.
        code = path.read_text(encoding="utf-8")
        if "parts.append('bell_noise')" not in code:
            raise RuntimeError(f"{name} lacks the exact _bell_noise report suffix. "
                               "Install the supplied three compatible execution files.")
    for name in ("zne_cli.py", "zne_noise_extensions.py", "unitary_tools.py",
                 "local_folding_helpers.py", "U_config.py", "state_preparation.py"):
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"Required project file missing: {name}")
    # Test the command-line API and physics source without actually simulating.
    cli = (ROOT / "zne_cli.py").read_text(encoding="utf-8")
    ext = (ROOT / "zne_noise_extensions.py").read_text(encoding="utf-8")
    for flag in ("increasing_exponential_random_pauli", "constant_seeded_random_pauli",
                 "--bell-noise", "--constant-pauli-depth", "--noise-model") :
        if flag not in cli:
            raise RuntimeError(f"Installed zne_cli.py does not support {flag}")
    if '--legacy-eagle-report-names' not in cli:
        raise RuntimeError('Installed zne_cli.py must support legacy Eagle report naming')
    if 'USE_LEGACY_EAGLE_REPORT_NAMES' not in (ROOT / 'ZNE_depth_reduced_local.py').read_text(encoding='utf-8'):
        raise RuntimeError('Installed local simulator cannot produce historical Ver2 Eagle filenames')
    for token in ("class ExtendedNoiseController", "constant_seeded_random_pauli",
                  "exponential_random_pauli", "append_bell_noise",
                  "get_pauli_initial_probabilities", "nighthawk_r2") :
        if token not in ext:
            raise RuntimeError(f"Installed zne_noise_extensions.py missing {token}")


def command(job: Job) -> list[str]:
    cmd = [
        sys.executable, "-u", str(ROOT / cfg.METHOD_FILES[job.method]),
        "--machine", cfg.MACHINE,
        "--backend", cfg.BACKEND,
        "--noise", job.noise,
        "--noise-model", job.noise_model,
        "--num-qubits", str(job.num_qubits),
        "--unitary", job.unitary,
        "--depth", str(job.depth),
        "--fold-factors", *map(str, job.fold_factors),
        "--shots", str(cfg.SHOTS),
        "--base-seed", str(cfg.BASE_SEED),
        "--gate-weight-1q", str(cfg.GATE_WEIGHT_1Q),
        "--gate-weight-2q", str(cfg.GATE_WEIGHT_2Q),
        "--report-directory", str(job.output_directory),
    ]
    if job.noise == "constant_seeded_random_pauli":
        cmd += ["--constant-pauli-depth", str(cfg.CONSTANT_PAULI_DEPTH)]
    if job.bell_noise:
        cmd += ["--bell-noise"]
    if job.legacy_report_name:
        cmd += ["--legacy-eagle-report-names"]
    if job.method == "local":
        cmd += ["--max-simulator-physical-qubits", str(cfg.MAX_SIMULATOR_PHYSICAL_QUBITS)]
    return cmd


def target_report_exists(job: Job) -> bool:
    """ONLY check the EXACT JSON filename in THIS job's output directory.

    IMPORTANT: Never inspect another report directory or a legacy filename.
    The configured report output path is the sole authority for skip decisions.
    """
    return job.report_path.is_file()


def validate_report(job: Job) -> tuple[bool, str]:
    """Validate ONLY the exact expected report file before marking SKIP.

    Missing exact file -> run the task regardless of reports elsewhere.
    Invalid exact file -> fail closed instead of overwriting it.
    """
    p = job.report_path
    if not p.is_file():
        return False, "missing"
    try:
        report = json.loads(p.read_text(encoding="utf-8"))
        ex = report["execution"]
        noise = report["noise"]
        weights = report["weights"]
        expect = report["expectation_values"]
        assert report["method"] == job.method
        assert int(report["num_qubits"]) == job.num_qubits
        assert report["unitary_type"] == job.unitary
        assert int(report["depth"]) == job.depth
        assert tuple(report["fold_factors"]) == job.fold_factors
        assert isinstance(expect, list) and len(expect) == len(job.fold_factors)
        assert all(math.isfinite(float(value)) for value in expect)
        assert ex["filename_tag"] == job.report_tag
        assert ex["backend_kind"] == "aer_simulator"
        assert ex["gpu_enabled"] is True
        assert int(ex["shots"]) == cfg.SHOTS
        assert math.isclose(float(weights["w1_one_qubit"]), cfg.GATE_WEIGHT_1Q, abs_tol=1e-12)
        assert math.isclose(float(weights["w2_two_qubit"]), cfg.GATE_WEIGHT_2Q, abs_tol=1e-12)
        # Circuit folding has no Bell preparation/measurement gates. Its report
        # correctly records bell_gate_noise_enabled=False even when the Bell
        # *batch stage* was requested (the filename still carries _bell_noise).
        # Global/local do contain Bell gates and record the requested flag.
        expected_bell_applied = bool(job.bell_noise and job.method != "circuit_folding")
        assert bool(noise.get("bell_gate_noise_enabled", False)) == expected_bell_applied
        expected_scaling = {
            "increasing_random_pauli": "linear_capped",
            "increasing_exponential_random_pauli": "exponential_capped",
            "constant_seeded_random_pauli": "constant_seeded_pauli_from_linear",
        }[job.noise]
        if job.legacy_report_name and 'depth_scaling_model' not in noise:
            # Historical Ver2 Eagle schema=4 predates explicit scaling metadata.
            # Its original linear-scaling configuration is verified below.
            assert expected_scaling == 'linear_capped'
            assert noise.get('selected_custom_noise_family') == 'depth_dependent'
        else:
            assert noise["depth_scaling_model"] == expected_scaling
        assert noise.get("hardware_inspired_noise_model", "eagle_r3") == job.noise_model
        if job.noise == "constant_seeded_random_pauli":
            config = noise["constant_seeded_random_pauli_configuration"]
            assert int(config["constant_reference_depth"]) == cfg.CONSTANT_PAULI_DEPTH
            assert "constant_seeded_random_pauli" in noise["active_custom_noise_types"]
        else:
            config = noise["depth_dependent_noise_configuration"]
            assert "depth_dependent_random_pauli" in noise["active_custom_noise_types"]
        assert int(config["base_seed"]) == cfg.BASE_SEED
        if job.legacy_report_name:
            # Fail closed if an unrelated historical report happens to share
            # the expected filename. The linear noise parameters must match.
            assert math.isclose(float(config['reference_depth']), 100.0, abs_tol=1e-12)
            assert math.isclose(float(config['noise_multiplier']), 3.0, abs_tol=1e-12)
            assert math.isclose(float(config['maximum_noise_multiplier']), 4.0, abs_tol=1e-12)
            assert math.isclose(float(config['cap_depth']), 150.0, abs_tol=1e-12)
        actual_init = config["initial_parameters"]
        expected_init = cfg.INITIAL_PAULI_PROBABILITIES[job.noise_model]
        for key in ("p_1q", "p_2q"):
            assert math.isclose(float(actual_init[key]), float(expected_init[key]),
                                rel_tol=1e-9, abs_tol=1e-14), key
        recorded_init = noise.get("initial_pauli_error_probabilities")
        if recorded_init is not None or job.noise_model == "nighthawk_r2":
            assert isinstance(recorded_init, dict)
            for key in ("p_1q", "p_2q"):
                assert math.isclose(float(recorded_init[key]), float(expected_init[key]),
                                    rel_tol=1e-9, abs_tol=1e-14), key
        # A syntactically valid, partially populated JSON is NOT a finished
        # report. Only the exact complete main-script schema can be skipped.
        counts = report["gate_counts"]
        cases = report["cases"]
        for phase in ("before_transpilation", "after_transpilation"):
            data = counts[phase]
            for label in ("N_1q", "N_2q", "N_weighted"):
                values = data[label]
                assert isinstance(values, list) and len(values) == len(job.fold_factors)
                assert all(math.isfinite(float(v)) and float(v) >= 0 for v in values)
            for n1, n2, weighted in zip(data["N_1q"], data["N_2q"], data["N_weighted"]):
                assert math.isclose(float(weighted),
                                    cfg.GATE_WEIGHT_1Q*float(n1)+cfg.GATE_WEIGHT_2Q*float(n2),
                                    rel_tol=1e-10, abs_tol=1e-7)
        for key in ("before_weighted", "before_two_qubit", "after_weighted", "after_two_qubit"):
            case = cases[key]
            assert isinstance(case["N"], list) and len(case["N"]) == len(job.fold_factors)
            assert all(math.isfinite(float(v)) for v in case["N"])
            for fitting in ("linear", "exponential"):
                assert isinstance(case[fitting], dict) and isinstance(case[fitting]["success"], bool)
        if job.bell_noise and job.method in ("global", "local"):
            bell_report = config["bell_noise_channel_counts"]
            assert sum(int(v) for v in bell_report.values()) > 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AssertionError) as exc:
        return False, f"invalid or incompatible JSON: {type(exc).__name__}: {exc}"
    return True, "valid"


def split_existing(jobs: list[Job]) -> tuple[list[Job], list[Job]]:
    """No cross-directory/legacy lookup. Check only job.report_path."""
    skipped, pending, bad = [], [], []
    for job in jobs:
        if target_report_exists(job):
            okay, reason = validate_report(job)
            if okay:
                skipped.append(job)
            else:
                bad.append(f"{job.report_path}: {reason}")
        else:
            pending.append(job)
    if bad:
        raise RuntimeError("Found existing incompatible/corrupted report(s) AT THE EXACT OUTPUT PATH. "
                           "Reports are NEVER overwritten.\n"
                           + "\n".join("  " + line for line in bad[:30])
                           + (f"\n... plus {len(bad) - 30} more" if len(bad) > 30 else ""))
    return skipped, pending


def record(job: Job, status: str, **extra: Any) -> dict[str, Any]:
    data = asdict(job)
    data.update({"status": status, "report_path": str(job.report_path), **extra})
    return data


def atomic_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def acquired_lock() -> Any:
    handle = (ROOT / ".zne_priority_batch.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("Another priority runner is already active in this project.")
    return handle


def check_other_project_jobs() -> None:
    """Prevents launching a new 8-GPU batch atop another active experiment."""
    monitored = {
        "run_experiments.py", "run_experiments_2.py", "run_experiments_ver2.py",
        "run_experiments_gpu2_local45.py", "run_experiments_priority.py",
        *cfg.METHOD_FILES.values(),
    }
    conflicts = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("The GPU batch launcher needs Ubuntu /proc.")
    for item in proc.iterdir():
        if not item.name.isdecimal() or int(item.name) == os.getpid():
            continue
        try:
            argv = [v.decode("utf-8", "replace") for v in (item / "cmdline").read_bytes().split(b"\0") if v]
            if not argv:
                continue
            names = {Path(arg).name for arg in argv if arg.endswith(".py")}
            if not names.intersection(monitored):
                continue
            # Concurrent --status / --dry-run is allowed.
            if names == {"run_experiments_priority.py"} and "--run" not in argv:
                continue
            cwd = (item / "cwd").resolve()
            if cwd == ROOT:
                conflicts.append(f"PID={item.name} " + " ".join(argv)[:180])
        except (FileNotFoundError, OSError, PermissionError, ProcessLookupError):
            pass
    if conflicts:
        raise RuntimeError("Other ZNE jobs are already active in this project. "
                           "Stop the old batch/workers before launching the new one:\n"
                           + "\n".join("  " + line for line in conflicts[:20]))


def gpu_preflight() -> None:
    script = ("from qiskit_aer import AerSimulator; "
              "devices=tuple(AerSimulator().available_devices()); "
              "assert 'GPU' in devices, f'GPU not available: {devices}'; "
              "print(devices)")
    for gpu_id in GPU_IDS:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
        proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode:
            raise RuntimeError(f"CUDA GPU {gpu_id} preflight failed: {proc.stdout}")
        message(f"GPU {gpu_id}: Aer CUDA PASS")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def display_job(job: Job) -> str:
    return (f"P{job.priority} {job.stage} [{job.noise_model}] | R{job.round} | {job.num_qubits}q "
            f"{job.unitary} depth={job.depth} {job.method} "
            f"factors={list(job.fold_factors)} | folder={job.directory_name}")


def dry_run(jobs: list[Job], show_all: bool, noise_model_selection: str) -> None:
    skipped, pending = split_existing(jobs)
    message(f"Project path: {ROOT} | selected profile(s): {noise_model_selection}")
    message(f"GPUs: {list(GPU_IDS)} | Total jobs: {len(jobs)} | valid reports: {len(skipped)} | pending: {len(pending)}")
    message("Priority: stage -> round (1..4) -> qubit (2..5) -> unitary -> method")
    message("SKIP RULE: exact target JSON path only; P15 uses historical names in zne_reports_ver2.")
    message("P1-P11 FAST (no 4/5q local); P12-P23 ONLY 4/5q local. Never block a free GPU.")
    message("No stage/round/qubit completion barriers. NO job timeouts. Fake backend disabled. Heron not scheduled.")
    message("Folds: 2/3/4q=[1,3,5,7], 5q=[1,3,5]; width upper bounds 4q=28, 5q=25.")
    skips = {j.index for j in skipped}
    for stage in cfg.selected_stages(noise_model_selection):
        selection = [j for j in jobs if j.priority == stage["priority"]]
        done = sum(j.index in skips for j in selection)
        message(f"\nP{stage['priority']} {stage['label']} | {stage['directory']} | "
                f"noise={stage['noise']} | noise_model={stage['noise_model']} | bell={stage['bell_noise']} | "
                f"{len(selection)} jobs ({done} SKIP, {len(selection)-done} pending)")
        for round_number in cfg.ROUND_ORDER:
            per_round = [j for j in selection if j.round == round_number]
            summary = Counter(j.num_qubits for j in per_round)
            message("    R" + str(round_number) + " | " + ", ".join(
                f"{n}q:{summary.get(n,0)}" for n in cfg.QUBIT_ORDER))
        if selection:
            message(f"    first: {display_job(selection[0])}")
            message(f"    last : {display_job(selection[-1])}")
    if show_all:
        for j in jobs:
            flag = "SKIP" if j.index in skips else "PLAN"
            message(f"[{j.index:04d}/{j.total:04d}] {flag} | {display_job(j)}\n    "
                    + " ".join(command(j)))
    message("\nDry run complete. No simulations or reports created.")


def read_status() -> None:
    path = ROOT / cfg.LOG_DIRECTORY / "latest_state.json"
    if not path.exists():
        message("No priority batch state yet. Start a batch with --run.")
        return
    content = json.loads(path.read_text(encoding="utf-8"))
    pid = int(content.get("runner_pid", -1))
    alive = Path(f"/proc/{pid}").exists() if pid > 0 else False
    message(f"Runner PID={pid} | alive={alive} | noise_models={content.get('noise_model_selection')} | status={content.get('status')} | "
            f"started={content.get('started_at')} | updated={content.get('updated_at')}")
    message(f"Total={content.get('configured')} | valid skips={content.get('skipped')} | "
            f"succeeded={content.get('succeeded')} | failed={content.get('failed')} | "
            f"queued={content.get('pending')} | active={len(content.get('active', {}))}")
    message(f"Session logs: {content.get('session_directory')}")
    for gpu in sorted(GPU_IDS):
        info = content.get("active", {}).get(str(gpu))
        if info:
            elapsed = int(time.time() - info.get("started_epoch", time.time()))
            message(f"GPU {gpu}: {info['label']} | running {elapsed//3600}h {(elapsed//60)%60}m | "
                    f"output={info['report_directory']} | log={info['log']}")
        else:
            message(f"GPU {gpu}: idle")
    message("Completed/queued by priority:")
    for key, vals in content.get("by_priority", {}).items():
        message(f"  {key}: {vals}")


def execute_job(job: Job, gpu: int, logs: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    path = logs / f"gpu{gpu}_{job.slug}.log"
    start = time.monotonic()
    result: dict[str, Any] = {"log": str(path), "returncode": None, "elapsed": None}
    try:
        with path.open("w", encoding="utf-8") as out:
            out.write(f"PRIORITY={job.priority} STAGE={job.stage} ROUND={job.round} "
                      f"GPU={gpu} DIRECTORY={job.directory_name}\n")
            out.write("COMMAND: " + " ".join(command(job)) + "\n\n")
            out.flush()
            proc = subprocess.Popen(command(job), cwd=ROOT, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            # Important: an unlimited wait. Never terminate for elapsed time.
            result["returncode"] = proc.wait()
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
    result["elapsed"] = round(time.monotonic() - start, 2)
    good, reason = validate_report(job)
    result["report_valid"] = good
    result["verification"] = reason
    return result


def run_batch(jobs: list[Job], skipped: list[Job], pending: list[Job]) -> int:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_base = (ROOT / cfg.LOG_DIRECTORY).resolve()
    logs = log_base / f"run_{stamp}_{os.getpid()}"
    logs.mkdir(parents=True, exist_ok=False)
    main_log_state = log_base / "latest_state.json"
    session_state = logs / "state.json"
    session_history = logs / "history.jsonl"
    session_results = logs / "results.json"
    stage_names = {str(s["priority"]): str(s["label"])
                   for s in cfg.STAGES if any(j.priority == s["priority"] for j in jobs)}
    counts = {f"P{k} {name}": {"total": 0, "skipped": 0, "succeeded": 0,
                               "failed": 0, "running": 0, "pending": 0}
              for k, name in stage_names.items()}
    for job in jobs:
        counts[f"P{job.priority} {job.stage}"]["total"] += 1
    for job in skipped:
        counts[f"P{job.priority} {job.stage}"]["skipped"] += 1
    for job in pending:
        counts[f"P{job.priority} {job.stage}"]["pending"] += 1
    state = {
        "runner_pid": os.getpid(), "status": "running", "started_at": now_iso(),
        "updated_at": now_iso(), "session_directory": str(logs),
        "configured": len(jobs), "skipped": len(skipped), "succeeded": 0,
        "noise_model_selection": ("both" if len({j.noise_model for j in jobs}) > 1
                                  else jobs[0].noise_model if jobs else None),
        "failed": 0, "pending": len(pending), "active": {},
        "gpu_ids": list(GPU_IDS), "by_priority": counts,
        "policy": "one global priority queue; no round/qubit/stage barriers; no job timeout",
    }
    results = [record(job, "skipped", gpu=None, seconds=0) for job in skipped]
    pending_queue = deque(pending)
    free = deque(GPU_IDS)
    running = {}
    state_lock = threading.RLock()
    stop_heartbeat = threading.Event()

    def save_state() -> None:
        # Serialize status writes across the scheduler and heartbeat thread.
        with state_lock:
            state["updated_at"] = now_iso()
            snap = json.loads(json.dumps(state))
            atomic_json(main_log_state, snap)
            atomic_json(session_state, snap)

    def event(action: str, job: Job, gpu: int | None, **details: Any) -> None:
        entry = {"timestamp": now_iso(), "event": action, "gpu": gpu,
                 "priority": job.priority, "stage": job.stage, "noise_model": job.noise_model,
                 "round": job.round,
                 "num_qubits": job.num_qubits, "method": job.method,
                 "unitary": job.unitary, "depth": job.depth,
                 "report_directory": job.directory_name, **details}
        with session_history.open("a", encoding="utf-8") as log:
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def heartbeat() -> None:
        while not stop_heartbeat.wait(float(cfg.HEARTBEAT_SECONDS)):
            with state_lock:
                active = [(gpu, info["label"]) for gpu, info in state["active"].items()]
                remain = state["pending"]
                finished = state["succeeded"]
            save_state()
            message(f"HEARTBEAT | pending={remain} success={finished} | "
                    + ("; ".join(f"GPU{gpu}:{label}" for gpu, label in sorted(active))
                       if active else "no active jobs"))

    save_state()
    atomic_json(session_results, results)
    ticker = threading.Thread(target=heartbeat, daemon=True)
    ticker.start()
    message("SKIP RULE: exact target JSON path ONLY; P15 = historical zne_reports_ver2 filenames.")
    message(f"START: {len(jobs)} total, {len(skipped)} existing, {len(pending)} pending; "
            f"GPUs {list(GPU_IDS)}; logs {logs}")
    # One executor for ALL selected stages. Each GPU can run a different stage.
    try:
        with ThreadPoolExecutor(max_workers=len(GPU_IDS)) as executor:
            while pending_queue or running:
                while pending_queue and free:
                    job = pending_queue.popleft()
                    # Sync may add a report between initial scan and dispatch.
                    if target_report_exists(job):
                        okay, why = validate_report(job)
                        if not okay:
                            raise RuntimeError(f"Existing target report changed before dispatch: {job.report_path}: {why}")
                        results.append(record(job, "skipped", gpu=None, seconds=0))
                        with state_lock:
                            state["pending"] -= 1
                            state["skipped"] += 1
                            c = counts[f"P{job.priority} {job.stage}"]
                            c["pending"] -= 1
                            c["skipped"] += 1
                        event("skip", job, None)
                        continue
                    gpu = free.popleft()
                    log_path = logs / f"gpu{gpu}_{job.slug}.log"
                    with state_lock:
                        state["active"][str(gpu)] = {
                            "label": display_job(job),
                            "priority": job.priority, "stage": job.stage,
                            "noise_model": job.noise_model,
                            "round": job.round, "num_qubits": job.num_qubits,
                            "report_directory": job.directory_name,
                            "report_filename": job.report_name,
                            "log": str(log_path), "started_epoch": time.time(),
                        }
                        state["pending"] -= 1
                        c = counts[f"P{job.priority} {job.stage}"]
                        c["pending"] -= 1
                        c["running"] += 1
                    future = executor.submit(execute_job, job, gpu, logs)
                    running[future] = (job, gpu)
                    event("start", job, gpu)
                    message(f"[{job.index:04d}/{job.total:04d}] START GPU={gpu} | {display_job(job)}")
                    save_state()
                if not running:
                    continue
                # Wait for ANY completion, not for all jobs within a phase.
                finished, _ = wait(set(running), return_when=FIRST_COMPLETED)
                for task in finished:
                    job, gpu = running.pop(task)
                    free.append(gpu)
                    try:
                        outcome = task.result()
                    except Exception as exc:
                        outcome = {"returncode": None, "report_valid": False,
                                   "exception": repr(exc), "elapsed": None,
                                   "verification": "thread failure", "log": None}
                    okay = (outcome.get("returncode") == 0 and outcome.get("report_valid"))
                    status = "succeeded" if okay else "failed"
                    results.append(record(job, status, gpu=gpu, **outcome))
                    atomic_json(session_results, results)
                    with state_lock:
                        state["active"].pop(str(gpu), None)
                        state[status] += 1
                        c = counts[f"P{job.priority} {job.stage}"]
                        c["running"] -= 1
                        c[status] += 1
                    event(status, job, gpu, **outcome)
                    message(f"[{job.index:04d}/{job.total:04d}] {status.upper()} GPU={gpu} | "
                            f"{display_job(job)} | seconds={outcome.get('elapsed')} "
                            f"| rc={outcome.get('returncode')} | report_ok={outcome.get('report_valid')}")
                    save_state()
    finally:
        stop_heartbeat.set()
        ticker.join()
        with state_lock:
            if state["active"] or state["pending"]:
                state["status"] = "interrupted_or_orphaned"
            else:
                state["status"] = "complete" if state["failed"] == 0 else "completed_with_failures"
        save_state()
        atomic_json(session_results, results)
    message(f"DONE | total={len(jobs)} skipped={state['skipped']} succeeded={state['succeeded']} "
            f"failed={state['failed']} | session={logs}")
    return 1 if state["failed"] else 0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="Strict dry run: validates the actual U_config/helper before printing the plan")
    modes.add_argument("--plan-only", action="store_true", help="Show job plan WITHOUT checking external U_config; not safe to launch from this alone")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--verify-environment", action="store_true", help="Verify the installed U_config and helper support all scheduled jobs (no GPUs)")
    parser.add_argument("--show-all", action="store_true", help="Print all selected commands in dry-run.")
    parser.add_argument(
        "--noise-model", choices=cfg.NOISE_MODEL_CHOICES, default="both",
        help="Select only Nighthawk r2, only Eagle r3, or both in the configured "
             "23-stage priority order (default: both). Heron is NOT scheduled.",
    )
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.status:
        read_status()
        return 0
    if cfg.BACKEND != "aer_simulator" or cfg.MACHINE != "gpu":
        raise RuntimeError("This plan must use GPU Aer simulator only, never Fake Backend")
    jobs = expand_jobs(args.noise_model)
    validate_plan(jobs)
    validate_execution_safety(jobs)
    if args.verify_environment:
        validate_files_and_methods(jobs)
        validate_unitary_runtime_compatibility(jobs)
        return 0
    if args.plan_only:
        dry_run(jobs, show_all=args.show_all, noise_model_selection=args.noise_model)
        message("PLAN ONLY: external U_config compatibility was NOT checked; run --dry-run before --run.")
        return 0
    # Strict --dry-run and actual --run both validate ALL unitary depths against
    # the installed GPU project's files; do not quietly hide an invalid plan.
    validate_files_and_methods(jobs)
    validate_unitary_runtime_compatibility(jobs)
    if args.dry_run or not args.run:
        dry_run(jobs, show_all=args.show_all, noise_model_selection=args.noise_model)
        return 0
    lock = acquired_lock()
    try:
        check_other_project_jobs()
        skipped, pending = split_existing(jobs)
        if any(j.legacy_report_name for j in jobs):
            # The user is RESUMING an existing Ver2 Eagle dataset (historically
            # 13/16 4q + 16/16 5q Local reports). A missing/misplaced dataset
            # must not silently start all 32 extremely expensive jobs.
            already = sum(j.legacy_report_name for j in skipped)
            if already < 29:
                raise RuntimeError(
                    f'Legacy Ver2 Eagle resume safety: only {already}/32 existing '
                    'valid local reports found in zne_reports_ver2 (expected at least 29). '
                    'Check report synchronization and original filenames before starting GPUs.'
                )
        for stage in cfg.selected_stages(args.noise_model):
            (ROOT / stage["directory"]).mkdir(exist_ok=True)
        message("GPU preflight...")
        gpu_preflight()
        # Recheck all folders before first job (e.g. copied reports from another GPU).
        skipped, pending = split_existing(jobs)
        return run_batch(jobs, skipped, pending)
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
