"""Shared command-line overrides for the three ZNE experiment scripts.

Running a main file without arguments preserves its editable in-file defaults.
The batch runner supplies explicit arguments, so it never rewrites the source
files between experiments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence


MACHINE_CHOICES = ("cpu", "gpu", "real_hardware")
BACKEND_CHOICES = (
    "aer_simulator",
    "fake_backend",
    "fake_statevector",
    "statevector",
)
NOISE_CHOICES = (
    "none",
    "constant_depolarizing",
    "constant_amplitude_damping",
    "constant_phase_damping",
    "constant_coherent_overrotation",
    "increasing_random_pauli",
    "increasing_amplitude_damping",
    "increasing_random_pauli_amplitude_damping",
)


@dataclass(frozen=True)
class CliArguments:
    machine: Optional[str]
    backend: Optional[str]
    noise: Optional[str]
    num_qubits: Optional[int]
    unitary: Optional[str]
    depth: Optional[int]
    report_directory: Optional[str]
    fake_backend_name: Optional[str]
    real_hardware_device_name: Optional[str]
    shots: Optional[int]
    base_seed: Optional[int]
    fold_factors: Optional[tuple[int, ...]]
    gate_weight_1q: Optional[float]
    gate_weight_2q: Optional[float]
    max_simulator_physical_qubits: Optional[int]


def _lower(value: str) -> str:
    return value.strip().lower()


def parse_cli_args(description: str) -> CliArguments:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--machine",
        type=_lower,
        choices=MACHINE_CHOICES,
        help="Execution machine: CPU/GPU for Aer, or a real IBM device.",
    )
    parser.add_argument(
        "--backend",
        type=_lower,
        choices=BACKEND_CHOICES,
        help="Simulation backend. Omit for real hardware.",
    )
    parser.add_argument(
        "--noise",
        type=_lower,
        choices=NOISE_CHOICES,
        help=(
            "Custom Aer noise mode. fake_backend, fake_statevector, and "
            "ideal statevector require 'none'."
        ),
    )
    parser.add_argument("--num-qubits", type=int)
    parser.add_argument("--unitary", type=str)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--report-directory", type=str)
    parser.add_argument("--fake-backend-name", type=str)
    parser.add_argument("--real-hardware-device-name", type=str)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument(
        "--fold-factors",
        nargs="+",
        type=int,
        help="Fold factors, for example: --fold-factors 1 3 5 7",
    )
    parser.add_argument("--gate-weight-1q", type=float)
    parser.add_argument("--gate-weight-2q", type=float)
    parser.add_argument(
        "--max-simulator-physical-qubits",
        type=int,
        help="Local-method resource guard override.",
    )

    namespace = parser.parse_args()
    fold_factors = None
    if namespace.fold_factors is not None:
        fold_factors = tuple(int(value) for value in namespace.fold_factors)
        if not fold_factors:
            parser.error("--fold-factors must not be empty.")
        if any(value < 1 or value % 2 == 0 for value in fold_factors):
            parser.error("Every fold factor must be a positive odd integer.")
        if tuple(sorted(set(fold_factors))) != fold_factors:
            parser.error("--fold-factors must be strictly increasing and unique.")

    for name in ("num_qubits", "depth", "shots", "base_seed"):
        value = getattr(namespace, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if (namespace.max_simulator_physical_qubits is not None
            and namespace.max_simulator_physical_qubits < 1):
        parser.error("--max-simulator-physical-qubits must be positive.")

    return CliArguments(
        machine=namespace.machine,
        backend=namespace.backend,
        noise=namespace.noise,
        num_qubits=namespace.num_qubits,
        unitary=namespace.unitary,
        depth=namespace.depth,
        report_directory=namespace.report_directory,
        fake_backend_name=namespace.fake_backend_name,
        real_hardware_device_name=namespace.real_hardware_device_name,
        shots=namespace.shots,
        base_seed=namespace.base_seed,
        fold_factors=fold_factors,
        gate_weight_1q=namespace.gate_weight_1q,
        gate_weight_2q=namespace.gate_weight_2q,
        max_simulator_physical_qubits=namespace.max_simulator_physical_qubits,
    )


def override(default, supplied):
    return default if supplied is None else supplied


def resolve_machine_flags(
    selected: Optional[str],
    *,
    default_use_real_hardware: bool,
    default_use_gpu: bool,
) -> tuple[bool, bool]:
    if selected is None:
        return bool(default_use_real_hardware), bool(default_use_gpu)
    return selected == "real_hardware", selected == "gpu"


def resolve_backend_flags(
    selected: Optional[str],
    *,
    use_real_hardware: bool,
    default_use_aer_simulator: bool,
    default_use_statevector: bool,
    default_use_fake_backend: bool,
    default_use_fake_statevector: bool = False,
) -> tuple[bool, bool, bool, bool]:
    """Resolve four mutually exclusive local-simulation backends.

    ``statevector`` remains the existing exact ideal/noiseless mode.
    ``fake_statevector`` keeps the fake-device noise model, forces Aer
    method="statevector", and returns finite-shot results from noisy pure-state
    trajectories.  It is therefore not an ideal statevector calculation.
    """
    if use_real_hardware:
        if selected is not None:
            raise ValueError(
                "Do not supply --backend when --machine real_hardware is selected."
            )
        return False, False, False, False

    if selected is None:
        return (
            bool(default_use_aer_simulator),
            bool(default_use_statevector),
            bool(default_use_fake_backend),
            bool(default_use_fake_statevector),
        )
    return (
        selected == "aer_simulator",
        selected == "statevector",
        selected == "fake_backend",
        selected == "fake_statevector",
    )


def resolve_noise_flags(
    selected: Optional[str],
    *,
    defaults: Sequence[bool],
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Return constant depol/AD/phase/coherent and increasing Pauli/AD flags."""
    if len(defaults) != 6:
        raise ValueError("Noise defaults must contain exactly six boolean values.")
    if selected is None:
        return tuple(bool(value) for value in defaults)  # type: ignore[return-value]

    flags = {
        "constant_depolarizing": False,
        "constant_amplitude_damping": False,
        "constant_phase_damping": False,
        "constant_coherent_overrotation": False,
        "increasing_random_pauli": False,
        "increasing_amplitude_damping": False,
    }
    if selected == "constant_depolarizing":
        flags["constant_depolarizing"] = True
    elif selected == "constant_amplitude_damping":
        flags["constant_amplitude_damping"] = True
    elif selected == "constant_phase_damping":
        flags["constant_phase_damping"] = True
    elif selected == "constant_coherent_overrotation":
        flags["constant_coherent_overrotation"] = True
    elif selected == "increasing_random_pauli":
        flags["increasing_random_pauli"] = True
    elif selected == "increasing_amplitude_damping":
        flags["increasing_amplitude_damping"] = True
    elif selected == "increasing_random_pauli_amplitude_damping":
        flags["increasing_random_pauli"] = True
        flags["increasing_amplitude_damping"] = True
    elif selected != "none":
        raise ValueError(f"Unsupported noise mode: {selected!r}.")

    return (
        flags["constant_depolarizing"],
        flags["constant_amplitude_damping"],
        flags["constant_phase_damping"],
        flags["constant_coherent_overrotation"],
        flags["increasing_random_pauli"],
        flags["increasing_amplitude_damping"],
    )
