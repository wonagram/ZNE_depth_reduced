"""Shared physical-depth decomposition, one-shot U compilation and noise policy.

This module is used by circuit folding, global depth reduction and local depth
reduction.  It deliberately compiles only operation plans generated for U.
Initial-state preparation, singlet preparation, Bell measurements and Pauli
post-corrections are never inspected or simplified.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np

from U_config import GateSpec, SelectedComponent


@dataclass(frozen=True)
class PrimitiveOp:
    name: str
    qubits: tuple[int, ...]
    angle: Optional[float] = None
    tag: str = ""
    macro_id: str = ""


@dataclass(frozen=True)
class NoiseMarker:
    # Noise is attached to the emitted primitive gate, not to the logical macro.
    physical_gate: str
    qubits: tuple[int, ...]
    direction: int = 1
    macro_id: str = ""


PlanItem = Union[PrimitiveOp, NoiseMarker]


_SELF_INVERSE = {"h", "x", "y", "z", "cx"}


def _p(name: str, qubits: Sequence[int], angle: Optional[float] = None,
       *, tag: str = "", macro_id: str = "") -> PrimitiveOp:
    return PrimitiveOp(name, tuple(int(q) for q in qubits), angle, tag, macro_id)


def decompose_logical_gate(gate: GateSpec, *, macro_id: str = "") -> tuple[tuple[PrimitiveOp, ...], ...]:
    """Return physical-depth-one decomposition slices.

    A slice may contain several mutually disjoint one-qubit gates, but no qubit
    is acted on twice inside one slice.  Thus every returned slice has circuit
    depth one.
    """
    name = gate.name
    q = gate.qubits
    theta = gate.angle

    if name in {"rx", "ry", "rz"}:
        return ((_p(name, q, theta, tag="rotation", macro_id=macro_id),),)
    if name in {"x", "y", "z", "h", "s", "sx", "cx"}:
        tag = "h" if name == "h" else "logical"
        return ((_p(name, q, tag=tag, macro_id=macro_id),),)

    if len(q) != 2:
        raise ValueError(f"{name} requires two qubits.")
    control, target = q

    if name == "cz":
        # H(target) - CX(control,target) - H(target): physical depth 3.
        return (
            (_p("h", (target,), tag="basis_h", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("h", (target,), tag="basis_h", macro_id=macro_id),),
        )

    if theta is None:
        raise ValueError(f"{name} requires an angle.")

    if name == "crz":
        # RZ(theta/2) - CX - RZ(-theta/2) - CX: physical depth 4.
        return (
            (_p("rz", (target,), theta / 2.0, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), -theta / 2.0, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
        )

    if name == "crx":
        # H - RZ(theta/2) - CX - RZ(-theta/2) - CX - H: depth 6.
        return (
            (_p("h", (target,), tag="basis_h", macro_id=macro_id),),
            (_p("rz", (target,), theta / 2.0, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), -theta / 2.0, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("h", (target,), tag="basis_h", macro_id=macro_id),),
        )

    if name == "rzz":
        return (
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
        )

    if name == "rxx":
        # Parallel H basis layer, then CX-RZ-CX, then parallel H layer: depth 5.
        return (
            (
                _p("h", (control,), tag="basis_h", macro_id=macro_id),
                _p("h", (target,), tag="basis_h", macro_id=macro_id),
            ),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (
                _p("h", (control,), tag="basis_h", macro_id=macro_id),
                _p("h", (target,), tag="basis_h", macro_id=macro_id),
            ),
        )

    if name == "ryy":
        # Parallel RX basis layer, then CX-RZ-CX, then inverse basis layer: depth 5.
        return (
            (
                _p("rx", (control,), -math.pi / 2.0, tag="basis_ryy", macro_id=macro_id),
                _p("rx", (target,), -math.pi / 2.0, tag="basis_ryy", macro_id=macro_id),
            ),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (_p("rz", (target,), theta, tag="macro_rotation", macro_id=macro_id),),
            (_p("cx", (control, target), tag="macro_cx", macro_id=macro_id),),
            (
                _p("rx", (control,), math.pi / 2.0, tag="basis_ryy", macro_id=macro_id),
                _p("rx", (target,), math.pi / 2.0, tag="basis_ryy", macro_id=macro_id),
            ),
        )

    raise ValueError(f"Unsupported logical gate {name!r}.")


def _merge_component_slices(component: SelectedComponent) -> list[PrimitiveOp]:
    """Merge corresponding decomposition slices of disjoint logical gates.

    This is the allowed overlap rule: decomposed gates may share a counted layer
    only if their primitive operations are disjoint, so the physical circuit depth
    of that counted layer remains exactly one.
    """
    per_gate = []
    for gate_index, gate in enumerate(component.gates):
        macro_id = f"{component.label}:{gate_index}"
        slices = decompose_logical_gate(gate, macro_id=macro_id)
        if component.included_depth > len(slices):
            raise ValueError(
                f"{component.label}: selected {component.included_depth} slices, "
                f"but {gate.name} has only {len(slices)}."
            )
        per_gate.append(slices[:component.included_depth])

    result: list[PrimitiveOp] = []
    for slice_index in range(component.included_depth):
        physical_slice: list[PrimitiveOp] = []
        occupied: set[int] = set()
        for slices in per_gate:
            for op in slices[slice_index]:
                if occupied.intersection(op.qubits):
                    raise ValueError(
                        f"{component.label}: decomposition slice {slice_index + 1} "
                        "has physical depth greater than one. Gates sharing a "
                        "qubit must be placed in different counted layers."
                    )
                occupied.update(op.qubits)
                physical_slice.append(op)
        result.extend(physical_slice)
    return result

def _inverse_primitive(op: PrimitiveOp) -> list[PrimitiveOp]:
    if op.name in _SELF_INVERSE:
        return [op]
    if op.name in {"rx", "ry", "rz"}:
        if op.angle is None:
            raise ValueError(f"Missing angle for {op.name}.")
        return [PrimitiveOp(op.name, op.qubits, -op.angle, op.tag, op.macro_id)]
    if op.name == "s":
        # S^dagger = S^3.  This keeps the emitted primitive alphabet free of Sdg.
        return [PrimitiveOp("s", op.qubits, None, op.tag, op.macro_id) for _ in range(3)]
    if op.name == "sx":
        # SX^dagger = SX^3.  This keeps the emitted alphabet free of SXdg.
        return [PrimitiveOp("sx", op.qubits, None, op.tag, op.macro_id) for _ in range(3)]
    raise ValueError(f"Cannot invert primitive {op.name!r}.")


def component_plan(component: SelectedComponent, *, inverse: bool = False,
                   dress_cx: bool = False, include_noise: bool = True,
                   noise_direction: int = 1) -> list[PlanItem]:
    """Build one component plan; a cut decomposed macro stays one component.

    Custom simulator noise is inserted after EVERY emitted decomposition
    primitive (H/RX/RY/RZ/S/SX/Pauli/CX).  The extra global Y gates used to
    dress CX in even depth-reduced blocks are method-generated corrections, not
    decomposition primitives, so they do not receive an extra custom channel.
    """
    primitives = _merge_component_slices(component)
    if inverse:
        inverted: list[PrimitiveOp] = []
        for op in reversed(primitives):
            inverted.extend(_inverse_primitive(op))
        primitives = inverted

    if dress_cx:
        dressed: list[PrimitiveOp] = []
        for op in primitives:
            if op.name == "cx":
                a, b = op.qubits
                dressed.extend([
                    _p("y", (a,), tag="global_y", macro_id=op.macro_id),
                    _p("y", (b,), tag="global_y", macro_id=op.macro_id),
                    op,
                    _p("y", (a,), tag="global_y", macro_id=op.macro_id),
                    _p("y", (b,), tag="global_y", macro_id=op.macro_id),
                ])
            else:
                dressed.append(op)
        primitives = dressed

    items: list[PlanItem] = []
    for op in primitives:
        items.append(op)
        if include_noise and op.tag != "global_y":
            # Preserve the old sign convention for inverse rotation gates.
            direction = -1 if inverse and op.name in {"rx", "ry", "rz"} else 1
            items.append(NoiseMarker(
                physical_gate=op.name,
                qubits=op.qubits,
                direction=direction,
                macro_id=op.macro_id,
            ))
    return items

def _candidate_signature(op: PrimitiveOp):
    if op.name == "y" and op.tag == "global_y":
        return ("global_y",)
    if op.name == "h":
        return ("h",)
    if (op.name == "rx" and op.tag == "basis_ryy" and op.angle is not None
            and abs(abs(op.angle) - math.pi / 2.0) <= 1e-12):
        return ("basis_rx", float(op.angle))
    return None


def _candidate_pair_cancels(first, second) -> bool:
    if first[0] != second[0]:
        return False
    if first[0] in {"global_y", "h"}:
        return True
    if first[0] == "basis_rx":
        # Only inverse +/-pi/2 pairs cancel.  Two identical RX(-pi/2) gates
        # do NOT cancel and are deliberately retained.
        return abs(first[1] + second[1]) <= 1e-12
    return False


def compile_u_plan_once(items: Sequence[PlanItem]) -> tuple[list[PlanItem], dict[str, int]]:
    """Single stack pass over one freshly generated U block.

    Noise markers never block a cancellation and are never removed.  Gates on
    other wires are ignored.  Any non-candidate unitary touching a wire closes
    that wire's cancellation stack.  Because this function consumes a fresh
    logical plan before it is appended to a QuantumCircuit, it cannot touch
    state preparation, Bell preparation, measurements or corrections.
    """
    stacks: dict[int, list[tuple[int, tuple]]] = {}
    removed: set[int] = set()
    stats = {"global_y_pairs": 0, "h_pairs": 0, "basis_rx_pairs": 0}

    for index, item in enumerate(items):
        if isinstance(item, NoiseMarker):
            continue
        signature = _candidate_signature(item)
        if signature is not None and len(item.qubits) == 1:
            wire = item.qubits[0]
            stack = stacks.setdefault(wire, [])
            if stack and _candidate_pair_cancels(stack[-1][1], signature):
                previous, previous_signature = stack.pop()
                removed.add(previous)
                removed.add(index)
                key = {
                    "global_y": "global_y_pairs",
                    "h": "h_pairs",
                    "basis_rx": "basis_rx_pairs",
                }[signature[0]]
                stats[key] += 1
            else:
                stack.append((index, signature))
            continue

        # Any other unitary operation involving a wire blocks cancellations
        # across that operation on that wire.
        for wire in item.qubits:
            stacks.pop(wire, None)

    compiled = [item for i, item in enumerate(items) if i not in removed]
    return compiled, stats


def build_unitary_block_plan(components: Sequence[SelectedComponent], *,
                             inverse: bool = False, dress_cx: bool = False,
                             compile_cancellations: bool = True) -> tuple[list[PlanItem], dict[str, int]]:
    ordered = list(reversed(components)) if inverse else list(components)
    raw: list[PlanItem] = []
    for component in ordered:
        raw.extend(component_plan(
            component,
            inverse=inverse,
            dress_cx=dress_cx,
            include_noise=True,
            noise_direction=-1 if inverse else 1,
        ))
    if compile_cancellations:
        return compile_u_plan_once(raw)
    return raw, {"global_y_pairs": 0, "h_pairs": 0, "basis_rx_pairs": 0}


def _resolve_wire_map(wires: Union[Sequence, Mapping[int, object]], logical: int):
    return wires[logical]


def append_plan(qc, wires: Union[Sequence, Mapping[int, object]],
                items: Sequence[PlanItem], *, noise_controller=None,
                use_noise: bool = True) -> None:
    """Append a precompiled logical plan to a Qiskit circuit."""
    for item in items:
        if isinstance(item, NoiseMarker):
            if use_noise and noise_controller is not None:
                physical = [_resolve_wire_map(wires, q) for q in item.qubits]
                noise_controller.append(
                    qc, physical, item.physical_gate, item.direction
                )
            continue

        physical = [_resolve_wire_map(wires, q) for q in item.qubits]
        name = item.name
        if name == "h":
            qc.h(physical[0])
        elif name == "x":
            qc.x(physical[0])
        elif name == "y":
            qc.y(physical[0])
        elif name == "z":
            qc.z(physical[0])
        elif name == "s":
            qc.s(physical[0])
        elif name == "sx":
            qc.sx(physical[0])
        elif name == "rx":
            qc.rx(item.angle, physical[0])
        elif name == "ry":
            qc.ry(item.angle, physical[0])
        elif name == "rz":
            qc.rz(item.angle, physical[0])
        elif name == "cx":
            qc.cx(physical[0], physical[1])
        else:
            raise ValueError(f"Unsupported emitted primitive {name!r}.")


def append_unitary_block(qc, wires, components: Sequence[SelectedComponent], *,
                         inverse: bool = False, dress_cx: bool = False,
                         noise_controller=None, use_noise: bool = True,
                         compile_cancellations: bool = True) -> dict[str, int]:
    """Generate, compile exactly once, and append one U or U-dagger block."""
    plan, stats = build_unitary_block_plan(
        components,
        inverse=inverse,
        dress_cx=dress_cx,
        compile_cancellations=compile_cancellations,
    )
    append_plan(
        qc, wires, plan, noise_controller=noise_controller, use_noise=use_noise
    )
    return stats


def append_local_component_copy(qc, wire_map, component: SelectedComponent, *,
                                block_index: int, noise_controller=None,
                                use_noise: bool = True) -> None:
    """Append one atomic local-folding component on one logical copy.

    No custom H/Y/RX cancellation pass is applied in the local method.  The
    complete (possibly truncated) decomposed macro component is folded as one G_i.
    """
    items = component_plan(
        component,
        inverse=False,
        dress_cx=(block_index % 2 == 1),
        include_noise=True,
        noise_direction=1,
    )
    append_plan(
        qc, wire_map, items,
        noise_controller=noise_controller,
        use_noise=use_noise,
    )


ARTIFICIAL_NOISE_LABEL = "__zne_artificial_noise__"

_TWO_QUBIT_PAULI_LABELS = tuple(
    first + second
    for first in "IXYZ"
    for second in "IXYZ"
    if first + second != "II"
)

_DEPTH_IGNORED_OPERATION_NAMES = {
    "delay", "measure", "reset",
    "kraus", "superop", "quantum_channel", "pauli",
    "save_expectation_value", "save_statevector", "set_statevector",
}


def depth_noise_scale(depth: float, *, reference_depth: float = 100.0,
                      noise_multiplier: float = 3.0,
                      maximum_noise_multiplier: float = 4.0) -> float:
    """Capped linear scale used by the depth-dependent noise model.

    The scale is

        s(D) = min(
            1 + (noise_multiplier - 1) * D / reference_depth,
            maximum_noise_multiplier,
        ).

    With the project defaults, s(0)=1, s(100)=3, and s(D)=4 for D>=150.
    The cap prevents folded circuits from producing unphysical probabilities
    while preserving a monotone increase over the experimentally used depths.
    """
    depth = float(depth)
    reference_depth = float(reference_depth)
    noise_multiplier = float(noise_multiplier)
    maximum_noise_multiplier = float(maximum_noise_multiplier)
    if not np.isfinite(depth) or depth < 0.0:
        raise ValueError(f"Noise depth must be finite and nonnegative, got {depth}.")
    if not np.isfinite(reference_depth) or reference_depth <= 0.0:
        raise ValueError("REFERENCE_DEPTH must be positive and finite.")
    if not np.isfinite(noise_multiplier) or noise_multiplier < 1.0:
        raise ValueError("NOISE_MULTIPLIER must be finite and at least 1.")
    if (not np.isfinite(maximum_noise_multiplier)
            or maximum_noise_multiplier < noise_multiplier):
        raise ValueError(
            "MAX_NOISE_MULTIPLIER must be finite and no smaller than "
            "NOISE_MULTIPLIER."
        )
    uncapped = 1.0 + (noise_multiplier - 1.0) * depth / reference_depth
    return min(uncapped, maximum_noise_multiplier)


def depth_noise_parameters(depth: float, *, reference_depth: float = 100.0,
                           noise_multiplier: float = 3.0,
                           maximum_noise_multiplier: float = 4.0,
                           p1_initial: float = 0.000245,
                           p2_initial: float = 0.006970,
                           gamma1_initial: float = 0.000112173,
                           gamma2_initial: float = 0.002113149) -> dict[str, float]:
    """Return the four capped, depth-dependent noise parameters."""
    scale = depth_noise_scale(
        depth,
        reference_depth=reference_depth,
        noise_multiplier=noise_multiplier,
        maximum_noise_multiplier=maximum_noise_multiplier,
    )
    parameters = {
        "p_1q": float(p1_initial) * scale,
        "p_2q": float(p2_initial) * scale,
        "gamma_1q": float(gamma1_initial) * scale,
        "gamma_2q": float(gamma2_initial) * scale,
    }
    for name, value in parameters.items():
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                f"Depth-dependent parameter {name} is invalid at depth {depth}: {value}."
            )
        if value > 1.0:
            raise ValueError(
                f"Depth-dependent parameter {name} exceeds 1 at depth {depth}: "
                f"{value}. Reduce the initial value, multiplier, or circuit depth."
            )
    return parameters


def random_dirichlet_partition(rng: np.random.Generator, total: float,
                               count: int) -> np.ndarray:
    """Split ``total`` into ``count`` fresh positive Dirichlet weights."""
    total = float(total)
    count = int(count)
    if count <= 0:
        raise ValueError("Dirichlet partition count must be positive.")
    if not np.isfinite(total) or not 0.0 <= total <= 1.0:
        raise ValueError(f"Pauli error probability must be in [0, 1], got {total}.")
    weights = np.asarray(rng.dirichlet(np.ones(count)), dtype=float)
    probabilities = total * weights
    # Enforce the requested sum up to machine precision without changing the
    # independently generated Dirichlet proportions in any meaningful way.
    if count > 1:
        probabilities[-1] = total - float(np.sum(probabilities[:-1]))
    else:
        probabilities[0] = total
    if np.any(probabilities < -1e-15) or not np.isclose(
        float(np.sum(probabilities)), total, rtol=0.0, atol=1e-14
    ):
        raise RuntimeError("Generated Pauli probabilities do not sum to the requested total.")
    probabilities = np.maximum(probabilities, 0.0)
    return probabilities


def _mark_artificial_noise_instruction(instruction):
    """Label an inserted channel so depth/gate counters can exclude it."""
    try:
        instruction.label = ARTIFICIAL_NOISE_LABEL
    except Exception:
        # Qiskit versions differ slightly in Instruction label mutability.
        # The operation-name checks below still exclude standard Aer channels.
        pass
    return instruction


def _original_circuit_gate_depth(qc, target_wires: Sequence | None = None) -> int:
    """Depth of the current original-gate location, excluding noise channels.

    Depth is tracked separately on every physical wire.  Returning the depth on
    ``target_wires`` is essential because complete U copies on disjoint folding
    blocks are appended sequentially in Python but remain parallel in circuit
    depth.  Initial-state gates, Bell-state gates, method-generated basis
    changes, and previous logical gates on the relevant causal wires contribute;
    previously inserted artificial channels do not.  A barrier synchronizes its
    wires without adding a depth layer.
    """
    depths = [0] * int(qc.num_qubits)
    for item in qc.data:
        op = item.operation
        name = str(getattr(op, "name", ""))
        label = getattr(op, "label", None)
        qindices = [qc.find_bit(qubit).index for qubit in item.qubits]
        if not qindices:
            continue
        if label == ARTIFICIAL_NOISE_LABEL or name in _DEPTH_IGNORED_OPERATION_NAMES:
            continue
        if name == "barrier":
            synchronized = max(depths[index] for index in qindices)
            for index in qindices:
                depths[index] = synchronized
            continue

        gate_depth = max(depths[index] for index in qindices) + 1
        for index in qindices:
            depths[index] = gate_depth

    if target_wires is None:
        return max(depths, default=0)
    target_indices = [qc.find_bit(wire).index for wire in target_wires]
    if not target_indices:
        return 0
    return max(depths[index] for index in target_indices)


class NoiseController:
    """Insert either the historical constant noise or the new depth-dependent noise.

    The two custom-noise families are mutually exclusive:

    * Historical constant mode: depolarizing, amplitude damping, phase damping,
      and coherent overrotation retain their previous semantics.
    * Depth-dependent mode: independently randomized one- and two-qubit Pauli
      channels and/or deterministic amplitude damping use parameters evaluated
      at the original circuit depth of each error location.

    Decomposed macros still receive noise after every emitted primitive gate.
    The insertion locations and all folding logic are unchanged.
    """

    def __init__(self, *, use_depolarizing: bool,
                 use_amplitude_damping: bool,
                 use_phase_damping: bool,
                 use_coherent_overrotation: bool,
                 depolarizing_strength: float,
                 amplitude_damping_strength: float,
                 phase_damping_strength: float,
                 overrotation_epsilon: float,
                 use_increasing_random_pauli: bool = False,
                 use_increasing_amplitude_damping: bool = False,
                 base_seed: int = 150,
                 reference_depth: float = 100.0,
                 noise_multiplier: float = 3.0,
                 maximum_noise_multiplier: float = 4.0,
                 p1_initial: float = 0.000245,
                 p2_initial: float = 0.006970,
                 gamma1_initial: float = 0.000112173,
                 gamma2_initial: float = 0.002113149):
        from qiskit_aer.noise import (
            amplitude_damping_error,
            depolarizing_error,
            phase_damping_error,
        )

        # Historical constant-noise configuration.
        self.use_depolarizing = bool(use_depolarizing)
        self.use_amplitude_damping = bool(use_amplitude_damping)
        self.use_phase_damping = bool(use_phase_damping)
        self.use_coherent_overrotation = bool(use_coherent_overrotation)
        self.overrotation_epsilon = float(overrotation_epsilon)

        # New depth-dependent configuration.
        self.use_increasing_random_pauli = bool(use_increasing_random_pauli)
        self.use_increasing_amplitude_damping = bool(
            use_increasing_amplitude_damping
        )
        self.base_seed = int(base_seed)
        self.reference_depth = float(reference_depth)
        self.noise_multiplier = float(noise_multiplier)
        self.maximum_noise_multiplier = float(maximum_noise_multiplier)
        self.p1_initial = float(p1_initial)
        self.p2_initial = float(p2_initial)
        self.gamma1_initial = float(gamma1_initial)
        self.gamma2_initial = float(gamma2_initial)
        self.rng = np.random.default_rng(self.base_seed)

        self.constant_noise_active = any((
            self.use_depolarizing,
            self.use_amplitude_damping,
            self.use_phase_damping,
            self.use_coherent_overrotation,
        ))
        self.increasing_noise_active = any((
            self.use_increasing_random_pauli,
            self.use_increasing_amplitude_damping,
        ))
        if self.constant_noise_active and self.increasing_noise_active:
            raise ValueError(
                "Historical constant noise and depth-dependent noise cannot be "
                "enabled together in one NoiseController."
            )

        # Validate the requested reference behavior only when the new model is used.
        if self.increasing_noise_active:
            at_zero = depth_noise_parameters(
                0.0,
                reference_depth=self.reference_depth,
                noise_multiplier=self.noise_multiplier,
                maximum_noise_multiplier=self.maximum_noise_multiplier,
                p1_initial=self.p1_initial,
                p2_initial=self.p2_initial,
                gamma1_initial=self.gamma1_initial,
                gamma2_initial=self.gamma2_initial,
            )
            at_reference = depth_noise_parameters(
                self.reference_depth,
                reference_depth=self.reference_depth,
                noise_multiplier=self.noise_multiplier,
                maximum_noise_multiplier=self.maximum_noise_multiplier,
                p1_initial=self.p1_initial,
                p2_initial=self.p2_initial,
                gamma1_initial=self.gamma1_initial,
                gamma2_initial=self.gamma2_initial,
            )
            expected_initials = {
                "p_1q": self.p1_initial,
                "p_2q": self.p2_initial,
                "gamma_1q": self.gamma1_initial,
                "gamma_2q": self.gamma2_initial,
            }
            if self.noise_multiplier == 1.0:
                cap_depth = (
                    0.0 if self.maximum_noise_multiplier == 1.0 else None
                )
            else:
                cap_depth = (
                    self.reference_depth
                    * (self.maximum_noise_multiplier - 1.0)
                    / (self.noise_multiplier - 1.0)
                )
            at_cap = (
                None if cap_depth is None
                else depth_noise_parameters(
                    cap_depth,
                    reference_depth=self.reference_depth,
                    noise_multiplier=self.noise_multiplier,
                    maximum_noise_multiplier=self.maximum_noise_multiplier,
                    p1_initial=self.p1_initial,
                    p2_initial=self.p2_initial,
                    gamma1_initial=self.gamma1_initial,
                    gamma2_initial=self.gamma2_initial,
                )
            )
            for name, initial in expected_initials.items():
                if not math.isclose(
                    at_zero[name], initial, rel_tol=0.0, abs_tol=1e-15
                ):
                    raise AssertionError(
                        f"{name} does not equal its initial value at D=0."
                    )
                expected_reference = initial * self.noise_multiplier
                if not math.isclose(
                    at_reference[name], expected_reference,
                    rel_tol=1e-12, abs_tol=1e-15,
                ):
                    raise AssertionError(
                        f"{name} does not equal NOISE_MULTIPLIER times its initial "
                        "value at REFERENCE_DEPTH."
                    )
                if at_cap is not None:
                    expected_cap = initial * self.maximum_noise_multiplier
                    if not math.isclose(
                        at_cap[name], expected_cap,
                        rel_tol=1e-12, abs_tol=1e-15,
                    ):
                        raise AssertionError(
                            f"{name} does not equal MAX_NOISE_MULTIPLIER times its "
                            "initial value at the cap depth."
                        )

        self.depolarizing_1q = (
            _mark_artificial_noise_instruction(
                depolarizing_error(float(depolarizing_strength), 1).to_instruction()
            )
            if self.use_depolarizing else None
        )
        self.depolarizing_2q = (
            _mark_artificial_noise_instruction(
                depolarizing_error(float(depolarizing_strength), 2).to_instruction()
            )
            if self.use_depolarizing else None
        )
        self.amplitude_1q = (
            _mark_artificial_noise_instruction(
                amplitude_damping_error(
                    float(amplitude_damping_strength), canonical_kraus=False
                ).to_instruction()
            )
            if self.use_amplitude_damping else None
        )
        self.phase_1q = (
            _mark_artificial_noise_instruction(
                phase_damping_error(
                    float(phase_damping_strength), canonical_kraus=False
                ).to_instruction()
            )
            if self.use_phase_damping else None
        )

        self._runtime = {
            "max_original_gate_depth": 0,
            "last_original_gate_depth": None,
            "random_pauli_1q_channels": 0,
            "random_pauli_2q_channels": 0,
            "increasing_amplitude_1q_channels": 0,
            "increasing_amplitude_2q_gate_locations": 0,
            "last_depth_dependent_parameters": None,
        }

    def _append_instruction(self, qc, instruction, wires) -> None:
        qc.append(_mark_artificial_noise_instruction(instruction), list(wires))

    def _depth_parameters(self, qc, wires: Sequence) -> tuple[int, dict[str, float]]:
        depth = _original_circuit_gate_depth(qc, wires)
        parameters = depth_noise_parameters(
            depth,
            reference_depth=self.reference_depth,
            noise_multiplier=self.noise_multiplier,
            maximum_noise_multiplier=self.maximum_noise_multiplier,
            p1_initial=self.p1_initial,
            p2_initial=self.p2_initial,
            gamma1_initial=self.gamma1_initial,
            gamma2_initial=self.gamma2_initial,
        )
        self._runtime["last_original_gate_depth"] = int(depth)
        self._runtime["max_original_gate_depth"] = max(
            int(self._runtime["max_original_gate_depth"]), int(depth)
        )
        self._runtime["last_depth_dependent_parameters"] = dict(parameters)
        return int(depth), parameters

    def _append_constant_1q_channels(self, qc, wire) -> None:
        if self.depolarizing_1q is not None:
            self._append_instruction(qc, self.depolarizing_1q, [wire])
        if self.amplitude_1q is not None:
            self._append_instruction(qc, self.amplitude_1q, [wire])
        if self.phase_1q is not None:
            self._append_instruction(qc, self.phase_1q, [wire])

    def _append_random_pauli_1q(self, qc, wire, total_probability: float) -> None:
        from qiskit_aer.noise import pauli_error

        probabilities = random_dirichlet_partition(
            self.rng, total_probability, 3
        )
        terms = [
            ("I", 1.0 - total_probability),
            ("X", float(probabilities[0])),
            ("Y", float(probabilities[1])),
            ("Z", float(probabilities[2])),
        ]
        self._append_instruction(qc, pauli_error(terms).to_instruction(), [wire])
        self._runtime["random_pauli_1q_channels"] += 1

    def _append_random_pauli_2q(self, qc, wires, total_probability: float) -> None:
        from qiskit_aer.noise import pauli_error

        probabilities = random_dirichlet_partition(
            self.rng, total_probability, len(_TWO_QUBIT_PAULI_LABELS)
        )
        terms = [("II", 1.0 - total_probability)]
        terms.extend(
            (label, float(probability))
            for label, probability in zip(_TWO_QUBIT_PAULI_LABELS, probabilities)
        )
        self._append_instruction(qc, pauli_error(terms).to_instruction(), wires)
        self._runtime["random_pauli_2q_channels"] += 1

    def _append_increasing_amplitude(self, qc, wires, gamma: float,
                                     *, two_qubit_location: bool) -> None:
        from qiskit_aer.noise import amplitude_damping_error

        instruction = amplitude_damping_error(
            float(gamma), canonical_kraus=False
        ).to_instruction()
        for wire in wires:
            self._append_instruction(qc, instruction, [wire])
            self._runtime["increasing_amplitude_1q_channels"] += 1
        if two_qubit_location:
            self._runtime["increasing_amplitude_2q_gate_locations"] += 1

    def _append_coherent(self, qc, wire, physical_gate: str, direction: int) -> None:
        if not self.use_coherent_overrotation:
            return

        # Preserve the previous simulator semantics: coherent overrotation is
        # appended as a QuantumError instruction, rather than as an ordinary
        # logical gate that a transpiler might merge with neighboring gates.
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        from qiskit_aer.noise import coherent_unitary_error

        epsilon = float(direction) * self.overrotation_epsilon
        x_like = {"rx", "x", "sx", "crx", "rxx"}
        y_like = {"ry", "y", "ryy", "h"}
        error_circuit = QuantumCircuit(1)
        if physical_gate in x_like:
            error_circuit.rx(epsilon, 0)
        elif physical_gate in y_like:
            error_circuit.ry(epsilon, 0)
        else:
            error_circuit.rz(epsilon, 0)

        instruction = coherent_unitary_error(Operator(error_circuit)).to_instruction()
        self._append_instruction(qc, instruction, [wire])

    def append(self, qc, wires: Sequence, physical_gate: str, direction: int = 1) -> None:
        wires = list(wires)
        if len(wires) not in (1, 2):
            raise ValueError("NoiseController supports one- and two-qubit primitives only.")
        if len(wires) == 2 and physical_gate != "cx":
            raise ValueError(
                f"Unexpected two-qubit primitive {physical_gate!r}; logical macros "
                "must be decomposed to one-qubit gates and CX before noise insertion."
            )

        depth_parameters = None
        if self.increasing_noise_active:
            _, depth_parameters = self._depth_parameters(qc, wires)

        if len(wires) == 1:
            self._append_constant_1q_channels(qc, wires[0])
            if self.use_increasing_random_pauli:
                self._append_random_pauli_1q(
                    qc, wires[0], depth_parameters["p_1q"]
                )
            if self.use_increasing_amplitude_damping:
                self._append_increasing_amplitude(
                    qc, wires, depth_parameters["gamma_1q"],
                    two_qubit_location=False,
                )
            self._append_coherent(qc, wires[0], physical_gate, direction)
            return

        # Historical constant CX policy, preserved exactly.
        if self.depolarizing_2q is not None:
            self._append_instruction(qc, self.depolarizing_2q, wires)
        if self.amplitude_1q is not None:
            self._append_instruction(qc, self.amplitude_1q, [wires[0]])
            self._append_instruction(qc, self.amplitude_1q, [wires[1]])
        if self.phase_1q is not None:
            self._append_instruction(qc, self.phase_1q, [wires[0]])
            self._append_instruction(qc, self.phase_1q, [wires[1]])

        if self.use_increasing_random_pauli:
            self._append_random_pauli_2q(qc, wires, depth_parameters["p_2q"])
        if self.use_increasing_amplitude_damping:
            self._append_increasing_amplitude(
                qc, wires, depth_parameters["gamma_2q"],
                two_qubit_location=True,
            )
        # Preserve the historical model: no coherent overrotation is attached to CX.

    def runtime_report(self) -> dict:
        """JSON-serializable configuration and runtime validation summary."""
        return {
            "depth_definition": (
                "Per-location structural depth on the gate's physical wires; disjoint "
                "blocks can remain parallel. Artificial noise channels, measurement, "
                "reset, and delay are excluded, and barriers synchronize without "
                "increasing depth."
            ),
            "base_seed": self.base_seed,
            "scaling_formula": (
                "s(D)=min(1+(noise_multiplier-1)*D/reference_depth, "
                "maximum_noise_multiplier); parameter(D)=parameter(0)*s(D)"
            ),
            "reference_depth": self.reference_depth,
            "noise_multiplier": self.noise_multiplier,
            "maximum_noise_multiplier": self.maximum_noise_multiplier,
            "cap_depth": (
                None if self.noise_multiplier == 1.0
                else self.reference_depth
                * (self.maximum_noise_multiplier - 1.0)
                / (self.noise_multiplier - 1.0)
            ),
            "initial_parameters": {
                "p_1q": self.p1_initial,
                "p_2q": self.p2_initial,
                "gamma_1q": self.gamma1_initial,
                "gamma_2q": self.gamma2_initial,
            },
            "increments_per_depth": {
                "p_1q": (
                    (self.noise_multiplier - 1.0)
                    * self.p1_initial / self.reference_depth
                ),
                "p_2q": (
                    (self.noise_multiplier - 1.0)
                    * self.p2_initial / self.reference_depth
                ),
                "gamma_1q": (
                    (self.noise_multiplier - 1.0)
                    * self.gamma1_initial / self.reference_depth
                ),
                "gamma_2q": (
                    (self.noise_multiplier - 1.0)
                    * self.gamma2_initial / self.reference_depth
                ),
            },
            "parameters_at_depth_0": depth_noise_parameters(
                0.0,
                reference_depth=self.reference_depth,
                noise_multiplier=self.noise_multiplier,
                maximum_noise_multiplier=self.maximum_noise_multiplier,
                p1_initial=self.p1_initial,
                p2_initial=self.p2_initial,
                gamma1_initial=self.gamma1_initial,
                gamma2_initial=self.gamma2_initial,
            ),
            "parameters_at_reference_depth": depth_noise_parameters(
                self.reference_depth,
                reference_depth=self.reference_depth,
                noise_multiplier=self.noise_multiplier,
                maximum_noise_multiplier=self.maximum_noise_multiplier,
                p1_initial=self.p1_initial,
                p2_initial=self.p2_initial,
                gamma1_initial=self.gamma1_initial,
                gamma2_initial=self.gamma2_initial,
            ),
            "maximum_parameters": {
                "p_1q": self.p1_initial * self.maximum_noise_multiplier,
                "p_2q": self.p2_initial * self.maximum_noise_multiplier,
                "gamma_1q": self.gamma1_initial * self.maximum_noise_multiplier,
                "gamma_2q": self.gamma2_initial * self.maximum_noise_multiplier,
            },
            "runtime": dict(self._runtime),
        }


def configure_aer_simulator_device(simulator, *, use_gpu: bool = False) -> str:
    """Select CPU or GPU execution for an AerSimulator instance.

    CPU is the default when ``use_gpu`` is False.  GPU mode requires a
    GPU-enabled Qiskit Aer build and a compatible NVIDIA/CUDA environment.
    The check is performed before circuit execution so an unavailable GPU
    produces a direct, readable error instead of failing deep inside a job.

    Returns the selected Aer device name (``"CPU"`` or ``"GPU"``).
    """
    requested = "GPU" if bool(use_gpu) else "CPU"

    available_method = getattr(simulator, "available_devices", None)
    available = None
    if callable(available_method):
        try:
            available = tuple(str(value).upper() for value in available_method())
        except Exception as error:
            if use_gpu:
                raise RuntimeError(
                    "GPU execution was requested, but Aer could not query the "
                    "available simulation devices. Check the GPU-enabled Qiskit "
                    "Aer installation and CUDA/NVIDIA driver environment."
                ) from error

    if use_gpu and available is not None and "GPU" not in available:
        displayed = ", ".join(available) if available else "none"
        raise RuntimeError(
            "USE_GPU=True, but this Qiskit Aer installation does not report a "
            f"GPU device. Available Aer devices: {displayed}. Install a "
            "GPU-enabled Qiskit Aer build on a compatible NVIDIA/CUDA machine, "
            "or set USE_GPU=False to use the CPU."
        )

    try:
        simulator.set_options(device=requested)
    except Exception as error:
        raise RuntimeError(
            f"Aer could not select device={requested!r}. Check that the installed "
            "Qiskit Aer build supports this device."
        ) from error

    return requested


def make_fake_aer_backend(fake_backend_name: str, *, seed_simulator: int = 150,
                           required_qubits: int | None = None,
                           use_gpu: bool = False):
    """Create a noisy Aer simulator from an IBM fake-backend snapshot.

    The fake snapshot supplies coupling, basis gates, readout properties and
    device noise.  Aer performs the execution so mid-circuit measurement and
    if_test control flow can be simulated.  No custom NoiseController channels
    should be appended when this backend mode is selected.  ``use_gpu`` only
    changes the machine used by Aer; it does not change the fake-device noise
    snapshot or circuit semantics.
    """
    if not isinstance(fake_backend_name, str) or not fake_backend_name.strip():
        raise ValueError("FAKE_BACKEND_NAME must be a non-empty fake-backend class name.")

    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider

    class_name = fake_backend_name.strip()
    backend_class = getattr(fake_provider, class_name, None)
    if backend_class is None:
        available = sorted(
            name for name in dir(fake_provider)
            if name.startswith("Fake") and name not in {"FakeBackendV2", "FakeProviderForBackendV2"}
        )
        raise ValueError(
            f"Unknown or unavailable FAKE_BACKEND_NAME={class_name!r}. "
            "Install a qiskit-ibm-runtime version that contains it, or choose one "
            f"of the available classes: {', '.join(available)}"
        )

    snapshot_backend = backend_class()
    if required_qubits is not None and snapshot_backend.num_qubits < int(required_qubits):
        raise ValueError(
            f"{class_name} has {snapshot_backend.num_qubits} qubits, but this run "
            f"needs at least {int(required_qubits)} physical qubits."
        )
    simulator = AerSimulator.from_backend(snapshot_backend)
    configure_aer_simulator_device(simulator, use_gpu=use_gpu)
    simulator.set_options(seed_simulator=int(seed_simulator))
    return snapshot_backend, simulator

# ============================================================
# Small exact matrices used by validation and the local checker
# ============================================================

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
_S = np.diag([1, 1j]).astype(complex)
_SX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)


def _one_qubit_matrix(op: PrimitiveOp) -> np.ndarray:
    if op.name == "h":
        return _H
    if op.name == "x":
        return _X
    if op.name == "y":
        return _Y
    if op.name == "z":
        return _Z
    if op.name == "s":
        return _S
    if op.name == "sx":
        return _SX
    if op.name in {"rx", "ry", "rz"}:
        pauli = {"rx": _X, "ry": _Y, "rz": _Z}[op.name]
        return math.cos(op.angle / 2.0) * _I - 1j * math.sin(op.angle / 2.0) * pauli
    raise ValueError(op.name)


def _single_on_n(matrix: np.ndarray, logical: int, n: int) -> np.ndarray:
    result = np.array([[1]], dtype=complex)
    for q in reversed(range(n)):
        result = np.kron(result, matrix if q == logical else _I)
    return result


def _cx_on_n(control: int, target: int, n: int) -> np.ndarray:
    dim = 1 << n
    indices = np.arange(dim)
    destinations = indices ^ (((indices >> control) & 1) << target)
    matrix = np.zeros((dim, dim), dtype=complex)
    matrix[destinations, indices] = 1.0
    return matrix


def primitive_plan_matrix(primitives: Iterable[PrimitiveOp], n: int) -> np.ndarray:
    result = np.eye(1 << n, dtype=complex)
    for op in primitives:
        if op.name == "cx":
            matrix = _cx_on_n(op.qubits[0], op.qubits[1], n)
        else:
            matrix = _single_on_n(_one_qubit_matrix(op), op.qubits[0], n)
        result = matrix @ result
    return result


def selected_component_matrix(component: SelectedComponent, n: int) -> np.ndarray:
    primitives = [
        item for item in component_plan(
            component, inverse=False, dress_cx=False, include_noise=False
        ) if isinstance(item, PrimitiveOp)
    ]
    return primitive_plan_matrix(primitives, n)


def selected_unitary_matrix(components: Sequence[SelectedComponent], n: int) -> np.ndarray:
    result = np.eye(1 << n, dtype=complex)
    for component in components:
        result = selected_component_matrix(component, n) @ result
    return result
