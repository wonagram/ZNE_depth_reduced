"""Deterministic logical-unitary configurations used by all three ZNE scripts.

Depth conventions used here follow the physical depth of the requested
decomposition.  Every counted layer has circuit depth one: operations may
share a counted layer only when they act on disjoint qubits.

* RX/RY/RZ, Pauli, H, S, SX and CX consume one selected depth slot.
* CZ consumes 3 slots: H - CX - H.
* RZZ consumes 3 slots: CX - RZ - CX.
* CRZ consumes 4 slots: RZ - CX - RZ - CX.
* CRX consumes 6 slots: H - RZ - CX - RZ - CX - H.
* RXX and RYY consume 5 slots because their two basis changes are parallel
  one-qubit layers around the three-layer CX-RZ-CX core.
* Several disjoint logical gates may share the same decomposition slices.
  If DEPTH ends inside any decomposed macro, the retained prefix remains one
  atomic component for local folding.

The historical U_GATES dictionary and all original theta variables are kept
verbatim.  The runtime random_type configuration preserves the exact historical
1-3-qubit prefix, intentionally regularizes the 4-qubit pattern into an 8-depth
repeat, and then extends each supported random_type circuit deterministically
with seeded random angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import copy
import math
import random
from typing import Iterable, Optional


# ============================================================
# ORIGINAL fixed U parameters - DO NOT CHANGE
# ============================================================
theta_x1 = 0.5
theta_y1 = 0.6
theta_z1 = 0.8
theta_x2 = 0.4
theta_y2 = 0.4
theta_z2 = 0.4
theta_x3 = 0.6
theta_y3 = 0.5
theta_z3 = 0.9
theta_x4 = 0.3
theta_y4 = 0.4
theta_z4 = 0.5
theta_x5 = 0.8
theta_y5 = 0.7
theta_z5 = 0.6
theta_x6 = 0.7
theta_y6 = 0.7
theta_z6 = 0.7


# ============================================================
# ORIGINAL U gate sequence - kept verbatim for random_type
# ============================================================
U_GATES = {
    1: [
        [("x", 0, theta_x1)],
        [("y", 0, theta_y1)],
        [("z", 0, theta_z1)],
        [("x", 0, theta_x2)],
        [("y", 0, theta_y2)],
        [("z", 0, theta_z2)],
        [("x", 0, theta_x3)],
        [("y", 0, theta_y3)],
        [("z", 0, theta_z3)],
        [("x", 0, theta_x4)],
        [("y", 0, theta_y4)],
        [("z", 0, theta_z4)],
        [("x", 0, theta_x5)],
        [("y", 0, theta_y5)],
        [("z", 0, theta_z5)],
        [("x", 0, theta_x6)],
        [("y", 0, theta_y6)],
        [("z", 0, theta_z6)],
    ],
    2: [
        [("x", 0, theta_x1), ("x", 1, theta_x1)],
        [("cx", 0, 1)],
        [("y", 0, theta_y1), ("y", 1, theta_y1)],
        [("cx", 1, 0)],
        [("z", 0, theta_z1), ("z", 1, theta_z1)],
        [("cx", 0, 1)],
        [("x", 0, theta_x2), ("x", 1, theta_x2)],
        [("cx", 1, 0)],
        [("y", 0, theta_y2), ("y", 1, theta_y2)],
        [("cx", 0, 1)],
        [("z", 0, theta_z2), ("z", 1, theta_z2)],
        [("cx", 1, 0)],
        [("x", 0, theta_x3), ("x", 1, theta_x3)],
        [("cx", 0, 1)],
        [("y", 0, theta_y3), ("y", 1, theta_y3)],
        [("cx", 1, 0)],
        [("z", 0, theta_z3), ("z", 1, theta_z3)],
        [("cx", 0, 1)],
    ],
    3: [
        [("x", 0, theta_x1), ("x", 1, theta_x1)],
        [("cx", 0, 1)],
        [("cx", 1, 2)],
        [("y", 1, theta_y1), ("y", 2, theta_y1)],
        [("cx", 2, 1)],
        [("cx", 0, 1)],
        [("z", 0, theta_z1), ("z", 2, theta_z1)],
        [("cx", 1, 2)],
        [("cx", 2, 1)],
        [("x", 0, theta_x2), ("x", 1, theta_x2)],
        [("cx", 0, 1)],
        [("cx", 1, 2)],
        [("y", 1, theta_y2), ("y", 2, theta_y2)],
        [("cx", 2, 1)],
        [("cx", 0, 1)],
        [("z", 0, theta_z2), ("z", 2, theta_z2)],
        [("cx", 1, 2)],
        [("cx", 2, 1)],
    ],
    4: [
        [("x", 0, theta_x1), ("x", 1, theta_x1),
         ("x", 2, theta_x1), ("x", 3, theta_x1)],
        [("cx", 0, 1), ("cx", 2, 3)],
        [("y", 0, theta_y1), ("y", 1, theta_y1),
         ("y", 2, theta_y1), ("y", 3, theta_y1)],
        [("cx", 1, 2)],
        [("z", 0, theta_z1), ("z", 1, theta_z1),
         ("z", 2, theta_z1), ("z", 3, theta_z1)],
        [("cx", 3, 2), ("cx", 1, 0)],
        [("x", 0, theta_x2), ("x", 1, theta_x2),
         ("x", 2, theta_x2), ("x", 3, theta_x2)],
        [("cx", 2, 1)],
        [("y", 0, theta_y2), ("y", 1, theta_y2),
         ("y", 2, theta_y2), ("y", 3, theta_y2)],
        [("cx", 0, 1), ("cx", 2, 3)],
        [("z", 0, theta_z2), ("z", 1, theta_z2),
         ("z", 2, theta_z2), ("z", 3, theta_z2)],
        [("cx", 1, 2)],
        [("x", 0, theta_x3), ("x", 1, theta_x3),
         ("x", 2, theta_x3), ("x", 3, theta_x3)],
        [("cx", 3, 2), ("cx", 1, 0)],
        [("y", 0, theta_y3), ("y", 1, theta_y3),
         ("y", 2, theta_y3), ("y", 3, theta_y3)],
        [("cx", 2, 1)],
    ],
}

# ============================================================
# Runtime fixed prefixes for random_type
# ============================================================
# Keep the historical dictionary above untouched so previously used 1-3 qubit
# circuits can be reproduced exactly.  Four-qubit random_type had not yet been
# used for experiments, so its runtime prefix is intentionally regularized.
RANDOM_TYPE_U_GATES = copy.deepcopy(U_GATES)

# 4q: exact 8-depth structural repeat.
# Rotation layers repeat X - Y - X - Z, while the original CNOT schedule repeats.
# The numerical angle values from the original first 16 layers are retained in
# their original rotation-slot order; only the rotation axes are regularized.
RANDOM_TYPE_U_GATES[4] = [
    [("x", 0, theta_x1), ("x", 1, theta_x1),
     ("x", 2, theta_x1), ("x", 3, theta_x1)],
    [("cx", 0, 1), ("cx", 2, 3)],
    [("y", 0, theta_y1), ("y", 1, theta_y1),
     ("y", 2, theta_y1), ("y", 3, theta_y1)],
    [("cx", 1, 2)],
    [("x", 0, theta_z1), ("x", 1, theta_z1),
     ("x", 2, theta_z1), ("x", 3, theta_z1)],
    [("cx", 3, 2), ("cx", 1, 0)],
    [("z", 0, theta_x2), ("z", 1, theta_x2),
     ("z", 2, theta_x2), ("z", 3, theta_x2)],
    [("cx", 2, 1)],

    [("x", 0, theta_y2), ("x", 1, theta_y2),
     ("x", 2, theta_y2), ("x", 3, theta_y2)],
    [("cx", 0, 1), ("cx", 2, 3)],
    [("y", 0, theta_z2), ("y", 1, theta_z2),
     ("y", 2, theta_z2), ("y", 3, theta_z2)],
    [("cx", 1, 2)],
    [("x", 0, theta_x3), ("x", 1, theta_x3),
     ("x", 2, theta_x3), ("x", 3, theta_x3)],
    [("cx", 3, 2), ("cx", 1, 0)],
    [("z", 0, theta_y3), ("z", 1, theta_y3),
     ("z", 2, theta_y3), ("z", 3, theta_y3)],
    [("cx", 2, 1)],
]


# ============================================================
# Typed logical configuration API
# ============================================================

@dataclass(frozen=True)
class GateSpec:
    """One logical gate before decomposition."""

    name: str
    qubits: tuple[int, ...]
    angle: Optional[float] = None


@dataclass(frozen=True)
class ComponentSpec:
    """One atomic local-folding component.

    depth is the physical decomposition depth consumed by this component.
    All logical gates in a component are disjoint, so corresponding primitive
    slices may share the same counted layer without making circuit depth > 1.
    """

    label: str
    depth: int
    gates: tuple[GateSpec, ...]
    family: str


@dataclass(frozen=True)
class SelectedComponent:
    """A component after truncating the full unitary at the selected DEPTH."""

    index: int
    label: str
    full_depth: int
    included_depth: int
    gates: tuple[GateSpec, ...]
    family: str
    depth_start: int
    depth_stop: int


# Generated non-random circuit families are extended far enough for the planned
# depth sweeps.  HEA circuits are generated only in COMPLETE repeating units:
# local RY -> local RZ -> odd-edge matching A -> even-edge matching B.
# For 2 qubits, matching B is empty.  Every generated HEA therefore ends on a
# two-qubit entangler layer, never on an extra final rotation layer.
NEW_TYPE_MIN_DEPTH = 128
HEA_CZ_RZZ_2Q_TARGET_DEPTH = 120
HEA_CZ_RZZ_3_TO_5Q_TARGET_DEPTH = 128
HEA_CRZ_2Q_TARGET_DEPTH = 144
HEA_CRZ_3_TO_5Q_TARGET_DEPTH = 120
MAX_SUPPORTED_QUBITS = 5

# random_type keeps the old fixed prefix and extends only the new tail.
# 3q uses 126 rather than 120 so the final 9-depth repeat is not cut.
RANDOM_TYPE_EXTENSION_SEED = 150
RANDOM_TYPE_TARGET_DEPTHS = {
    1: 120,
    2: 120,
    3: 126,
    4: 128,
}

_LOGICAL_GATE_DEPTHS = {
    "rx": 1, "ry": 1, "rz": 1,
    "x": 1, "y": 1, "z": 1, "h": 1, "s": 1, "sx": 1,
    "cx": 1,
    "cz": 3,
    "rzz": 3,
    "crz": 4,
    "rxx": 5,
    "ryy": 5,
    "crx": 6,
}
_ALLOWED_LOGICAL_GATES = {
    "rx", "ry", "rz",
    "x", "y", "z", "h", "s", "sx",
    "cx", "cz", "crx", "crz", "rxx", "ryy", "rzz",
}

UNITARY_METADATA = {
    "random_type": {
        "minimum_qubits": 1,
        "description": (
            "Historical fixed prefix extended deterministically with seeded random "
            "angles; 4q uses an 8-depth X-Y-X-Z/CNOT repeat."
        ),
    },
    "hea_cz": {
        "minimum_qubits": 2,
        "description": "Layered HEA with RY/RZ local rotations and CZ chain matchings.",
    },
    "hea_crx": {
        "minimum_qubits": 2,
        "description": "Layered HEA with RY/RZ local rotations and CRX chain matchings.",
    },
    "hea_crz": {
        "minimum_qubits": 2,
        "description": "Layered HEA with RY/RZ local rotations and CRZ chain matchings.",
    },
    "hea_rzz": {
        "minimum_qubits": 2,
        "description": "Layered HEA with RY/RZ local rotations and three-layer RZZ matchings.",
    },
    "qaoa": {
        "minimum_qubits": 1,
        "description": "H preparation, RZZ/RZ Ising cost, and RX mixer.",
    },
    "qaoa_xy": {
        "minimum_qubits": 2,
        "description": "Basis preparation, RZZ cost, then matching RXX and RYY XY mixer.",
    },
    "clifford_only": {
        "minimum_qubits": 1,
        "description": "Clifford scaffold using H/S/SX/Paulis and CZ, with all T gates omitted.",
    },
}

UNITARY_TYPES = tuple(UNITARY_METADATA)
UNITARY_ALIASES = {
    "hea": "hea_cz",
    "clifford": "clifford_only",
}

_TYPE_SEEDS = {
    "hea_cz": 4197,
    "hea_crx": 4294,
    "hea_crz": 4391,
    "hea_rzz": 4488,
    "qaoa": 4585,
    "qaoa_xy": 4682,
    "clifford_only": 4779,
}


def _gate(name: str, *qubits: int, angle: Optional[float] = None) -> GateSpec:
    return GateSpec(name=name, qubits=tuple(int(q) for q in qubits), angle=angle)


def logical_gate_depth(name: str) -> int:
    if name not in _ALLOWED_LOGICAL_GATES:
        raise ValueError(f"Unsupported logical gate {name!r}.")
    return _LOGICAL_GATE_DEPTHS[name]


def _component(label: str, gates: Iterable[GateSpec], family: str) -> ComponentSpec:
    gates = tuple(gates)
    if not gates:
        raise ValueError(f"Component {label!r} cannot be empty.")
    depths = {logical_gate_depth(gate.name) for gate in gates}
    if len(depths) != 1:
        raise ValueError(
            f"Component {label!r} mixes logical gates with different physical "
            "decomposition depths. Put them in separate U-config components."
        )
    return ComponentSpec(
        label=label,
        depth=depths.pop(),
        gates=gates,
        family=family,
    )


def _chain_edges(n: int) -> list[tuple[int, int]]:
    return [(q, q + 1) for q in range(n - 1)]


def _chain_matchings(n: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    return (
        [(q, q + 1) for q in range(0, n - 1, 2)],
        [(q, q + 1) for q in range(1, n - 1, 2)],
    )


def _fixed_rng(unitary_type: str, n: int) -> random.Random:
    return random.Random(_TYPE_SEEDS[unitary_type] + 1009 * n)


def _wide_angle(rng: random.Random) -> float:
    return rng.uniform(-0.95 * math.pi, 0.95 * math.pi)


def _small_angle(rng: random.Random) -> float:
    magnitude = rng.uniform(0.04 * math.pi, 0.22 * math.pi)
    return magnitude if rng.random() < 0.5 else -magnitude


def _component_depth_sum(components: Iterable[ComponentSpec]) -> int:
    return sum(component.depth for component in components)


def _validate_components(unitary_type: str, n: int,
                         components: Iterable[ComponentSpec]) -> tuple[ComponentSpec, ...]:
    result = tuple(components)
    if not result:
        raise ValueError(f"Unitary type {unitary_type!r} produced no components.")
    for j, component in enumerate(result):
        occupied: set[int] = set()
        if component.depth < 1:
            raise ValueError(f"{component.label}: component depth must be positive.")
        for gate in component.gates:
            if gate.name not in _ALLOWED_LOGICAL_GATES:
                raise ValueError(f"{component.label}: unsupported gate {gate.name!r}.")
            expected_arity = 2 if gate.name in {
                "cx", "cz", "crx", "crz", "rxx", "ryy", "rzz"
            } else 1
            if len(gate.qubits) != expected_arity:
                raise ValueError(
                    f"{component.label}: {gate.name} needs {expected_arity} qubits."
                )
            if len(set(gate.qubits)) != len(gate.qubits):
                raise ValueError(f"{component.label}: repeated qubit in {gate}.")
            if any(q < 0 or q >= n for q in gate.qubits):
                raise ValueError(f"{component.label}: invalid qubit in {gate}.")
            if occupied.intersection(gate.qubits):
                raise ValueError(
                    f"{component.label}: logical gates share a qubit, so their "
                    "decomposition slices cannot all remain physical depth one. "
                    "Split them into separate components."
                )
            occupied.update(gate.qubits)
            needs_angle = gate.name in {
                "rx", "ry", "rz", "crx", "crz", "rxx", "ryy", "rzz"
            }
            if needs_angle:
                if gate.angle is None or not math.isfinite(float(gate.angle)):
                    raise ValueError(f"{component.label}: finite angle required for {gate}.")
            elif gate.angle is not None:
                raise ValueError(f"{component.label}: {gate.name} must not carry an angle.")
            if logical_gate_depth(gate.name) != component.depth:
                raise ValueError(
                    f"{component.label}: gate depth and component depth disagree."
                )
        if component.family != unitary_type:
            raise ValueError(f"Component {j} has the wrong family tag.")
    return result


def _random_type_extension_rng(n: int) -> random.Random:
    """Dedicated deterministic RNG for only the added random_type tail."""
    return random.Random(RANDOM_TYPE_EXTENSION_SEED + 1009 * int(n))


def _random_type_extra_layer(n: int, layer_index: int,
                             rng: random.Random):
    """Return one added legacy-style layer for random_type.

    layer_index is zero-based over the complete runtime sequence.  Existing
    fixed-prefix layers are never regenerated, so changing the extension seed
    cannot alter previously completed 1-3q experiments.
    """
    if n == 1:
        axis = ("x", "y", "z")[layer_index % 3]
        return [(axis, 0, _wide_angle(rng))]

    if n == 2:
        # Odd 1-based layers: X/Y/Z rotations on both qubits.
        # Even 1-based layers: CNOT direction alternates globally forever.
        one_based = layer_index + 1
        if one_based % 2 == 1:
            rotation_slot = (one_based + 1) // 2
            axis = ("x", "y", "z")[(rotation_slot - 1) % 3]
            angle = _wide_angle(rng)
            return [(axis, 0, angle), (axis, 1, angle)]
        cx_slot = one_based // 2
        return [("cx", 0, 1)] if cx_slot % 2 == 1 else [("cx", 1, 0)]

    if n == 3:
        # Exact 9-depth repeat retained from the historical 3q sequence.
        position = layer_index % 9
        if position == 0:
            angle = _wide_angle(rng)
            return [("x", 0, angle), ("x", 1, angle)]
        if position == 1:
            return [("cx", 0, 1)]
        if position == 2:
            return [("cx", 1, 2)]
        if position == 3:
            angle = _wide_angle(rng)
            return [("y", 1, angle), ("y", 2, angle)]
        if position == 4:
            return [("cx", 2, 1)]
        if position == 5:
            return [("cx", 0, 1)]
        if position == 6:
            angle = _wide_angle(rng)
            return [("z", 0, angle), ("z", 2, angle)]
        if position == 7:
            return [("cx", 1, 2)]
        return [("cx", 2, 1)]

    if n == 4:
        # Exact 8-depth repeat requested for the new 4q experiments.
        # Rotation-axis order inside every repeat is X - Y - X - Z.
        position = layer_index % 8
        if position == 0:
            angle = _wide_angle(rng)
            return [("x", q, angle) for q in range(4)]
        if position == 1:
            return [("cx", 0, 1), ("cx", 2, 3)]
        if position == 2:
            angle = _wide_angle(rng)
            return [("y", q, angle) for q in range(4)]
        if position == 3:
            return [("cx", 1, 2)]
        if position == 4:
            angle = _wide_angle(rng)
            return [("x", q, angle) for q in range(4)]
        if position == 5:
            return [("cx", 3, 2), ("cx", 1, 0)]
        if position == 6:
            angle = _wide_angle(rng)
            return [("z", q, angle) for q in range(4)]
        return [("cx", 2, 1)]

    raise ValueError(
        "UNITARY_TYPE='random_type' supports only 1-4 qubits. "
        "No historical 5-qubit random_type pattern exists."
    )


def _legacy_random_components(n: int) -> tuple[ComponentSpec, ...]:
    if n not in RANDOM_TYPE_U_GATES:
        raise ValueError(
            "UNITARY_TYPE='random_type' supports only 1-4 qubits. "
            "The original U_config.py did not define a 5-qubit random_type."
        )

    layers = copy.deepcopy(RANDOM_TYPE_U_GATES[n])
    target_depth = int(RANDOM_TYPE_TARGET_DEPTHS[n])
    rng = _random_type_extension_rng(n)

    while len(layers) < target_depth:
        layers.append(_random_type_extra_layer(n, len(layers), rng))

    converted = []
    axis_map = {"x": "rx", "y": "ry", "z": "rz"}

    for j, layer in enumerate(layers):
        gates = []

        for legacy in layer:
            if legacy[0] == "cx":
                gates.append(_gate("cx", legacy[1], legacy[2]))
            else:
                gates.append(
                    _gate(
                        axis_map[legacy[0]],
                        legacy[1],
                        angle=float(legacy[2]),
                    )
                )

        converted.append(
            _component(
                f"random_type_G{j + 1}",
                gates,
                "random_type",
            )
        )

    return tuple(converted)

def _hea_target_depth(unitary_type: str, n: int) -> int:
    """Return the planned HEA maximum depth at a complete repeat boundary."""
    if unitary_type in {"hea_cz", "hea_rzz"}:
        return (
            HEA_CZ_RZZ_2Q_TARGET_DEPTH
            if n == 2
            else HEA_CZ_RZZ_3_TO_5Q_TARGET_DEPTH
        )
    if unitary_type == "hea_crz":
        return (
            HEA_CRZ_2Q_TARGET_DEPTH
            if n == 2
            else HEA_CRZ_3_TO_5Q_TARGET_DEPTH
        )
    return NEW_TYPE_MIN_DEPTH


def _generate_hea(unitary_type: str, n: int, entangler: str) -> tuple[ComponentSpec, ...]:
    """Generate complete alternating-edge HEA repeats.

    One repeat is

        RY(all) -> RZ(all) -> matching A -> matching B,

    with matching A = (0,1), (2,3), ... and matching B = (1,2),
    (3,4), ... .  Empty matchings are skipped, so for 2 qubits one repeat is
    simply RY -> RZ -> A.

    Importantly, there is NO extra final RY/RZ stage.  Every repeat and every
    allowed stopping boundary ends on the final two-qubit matching of that
    repeat.  This restores the original HEA generation order, preserving the
    deterministic seeded parameters of all previously generated prefixes while
    extending the sequence to the larger planned depths.
    """
    rng = _fixed_rng(unitary_type, n)
    match_a, match_b = _chain_matchings(n)
    components: list[ComponentSpec] = []
    target_depth = _hea_target_depth(unitary_type, n)
    block = 0

    while _component_depth_sum(components) < target_depth:
        components.append(_component(
            f"{unitary_type}_B{block}_RY",
            [_gate("ry", q, angle=_wide_angle(rng)) for q in range(n)],
            unitary_type,
        ))
        components.append(_component(
            f"{unitary_type}_B{block}_RZ",
            [_gate("rz", q, angle=_wide_angle(rng)) for q in range(n)],
            unitary_type,
        ))

        for matching_name, edges in (("A", match_a), ("B", match_b)):
            if not edges:
                continue
            gates = []
            for a, b in edges:
                angle = _wide_angle(rng) if entangler != "cz" else None
                gates.append(_gate(entangler, a, b, angle=angle))
            components.append(_component(
                f"{unitary_type}_B{block}_{matching_name}", gates, unitary_type
            ))

        block += 1

    total_depth = _component_depth_sum(components)
    if total_depth != target_depth:
        raise AssertionError(
            f"{unitary_type} with {n} qubits generated depth {total_depth}, "
            f"but target depth {target_depth} is not a complete repeat boundary."
        )

    return tuple(components)



def _basis_preparation(n: int, unitary_type: str) -> ComponentSpec:
    occupied = list(range(0, n, 2))
    return _component(
        f"{unitary_type}_basis_preparation",
        [_gate("x", q) for q in occupied],
        unitary_type,
    )



def _generate_qaoa(n: int) -> tuple[ComponentSpec, ...]:
    unitary_type = "qaoa"
    rng = _fixed_rng(unitary_type, n)
    edges = _chain_edges(n)
    components = [_component(
        f"{unitary_type}_H", [_gate("h", q) for q in range(n)], unitary_type
    )]
    round_index = 0
    while _component_depth_sum(components) < NEW_TYPE_MIN_DEPTH:
        gamma = _small_angle(rng)
        for edge_index, (a, b) in enumerate(edges):
            components.append(_component(
                f"{unitary_type}_R{round_index}_ZZ_{edge_index}",
                [_gate("rzz", a, b, angle=2.0 * gamma * rng.uniform(0.4, 1.2))],
                unitary_type,
            ))
        components.append(_component(
            f"{unitary_type}_R{round_index}_Z_fields",
            [_gate("rz", q, angle=2.0 * gamma * rng.uniform(-1.0, 1.0))
             for q in range(n)],
            unitary_type,
        ))
        beta = _small_angle(rng)
        components.append(_component(
            f"{unitary_type}_R{round_index}_X_mixer",
            [_gate("rx", q, angle=2.0 * beta * rng.uniform(0.7, 1.3))
             for q in range(n)],
            unitary_type,
        ))
        round_index += 1
    return tuple(components)


def _generate_qaoa_xy(n: int) -> tuple[ComponentSpec, ...]:
    unitary_type = "qaoa_xy"
    rng = _fixed_rng(unitary_type, n)
    edges = _chain_edges(n)
    match_a, match_b = _chain_matchings(n)
    components = [_basis_preparation(n, unitary_type)]
    round_index = 0
    while _component_depth_sum(components) < NEW_TYPE_MIN_DEPTH:
        gamma = _small_angle(rng)
        for edge_index, (a, b) in enumerate(edges):
            components.append(_component(
                f"{unitary_type}_R{round_index}_cost_{edge_index}",
                [_gate("rzz", a, b, angle=-gamma * rng.uniform(0.5, 1.5))],
                unitary_type,
            ))
        beta = _small_angle(rng)
        for matching_name, matching in (("A", match_a), ("B", match_b)):
            if not matching:
                continue
            components.append(_component(
                f"{unitary_type}_R{round_index}_{matching_name}_RXX",
                [_gate("rxx", a, b, angle=beta) for a, b in matching],
                unitary_type,
            ))
            components.append(_component(
                f"{unitary_type}_R{round_index}_{matching_name}_RYY",
                [_gate("ryy", a, b, angle=beta) for a, b in matching],
                unitary_type,
            ))
        round_index += 1
    return tuple(components)




def _generate_clifford_only(n: int) -> tuple[ComponentSpec, ...]:
    unitary_type = "clifford_only"
    rng = _fixed_rng(unitary_type, n)
    match_a, match_b = _chain_matchings(n)
    choices = ("h", "s", "sx", "x", "y", "z")
    components: list[ComponentSpec] = []
    sweep = 0
    while _component_depth_sum(components) < NEW_TYPE_MIN_DEPTH:
        word_length = 1 + rng.randrange(3)
        for word_position in range(word_length):
            components.append(_component(
                f"{unitary_type}_S{sweep}_word_{word_position}",
                [_gate(rng.choice(choices), q) for q in range(n)],
                unitary_type,
            ))
        for matching_name, matching in (("A", match_a), ("B", match_b)):
            if matching:
                components.append(_component(
                    f"{unitary_type}_S{sweep}_CZ_{matching_name}",
                    [_gate("cz", a, b) for a, b in matching],
                    unitary_type,
                ))
        components.append(_component(
            f"{unitary_type}_S{sweep}_final_H",
            [_gate("h", q) for q in range(n)],
            unitary_type,
        ))
        sweep += 1
    return tuple(components)


def normalize_unitary_type(unitary_type: str) -> str:
    if not isinstance(unitary_type, str):
        raise TypeError("UNITARY_TYPE must be a string.")
    normalized = unitary_type.strip().lower()
    normalized = UNITARY_ALIASES.get(normalized, normalized)
    if normalized not in UNITARY_METADATA:
        allowed = ", ".join(UNITARY_TYPES)
        raise ValueError(
            f"Unknown UNITARY_TYPE={unitary_type!r}. Available types: {allowed}."
        )
    return normalized


@lru_cache(maxsize=None)
def _cached_unitary_components(unitary_type: str, n: int) -> tuple[ComponentSpec, ...]:
    unitary_type = normalize_unitary_type(unitary_type)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_SUPPORTED_QUBITS:
        raise ValueError(
            f"NUM_U_QUBITS must be an integer from 1 to {MAX_SUPPORTED_QUBITS}; got {n}."
        )
    minimum = int(UNITARY_METADATA[unitary_type]["minimum_qubits"])
    if n < minimum:
        raise ValueError(
            f"UNITARY_TYPE={unitary_type!r} requires at least {minimum} qubits; "
            f"NUM_U_QUBITS={n} is not supported."
        )

    if unitary_type == "random_type":
        components = _legacy_random_components(n)
    elif unitary_type.startswith("hea_"):
        components = _generate_hea(unitary_type, n, unitary_type.removeprefix("hea_"))
    elif unitary_type == "qaoa":
        components = _generate_qaoa(n)
    elif unitary_type == "qaoa_xy":
        components = _generate_qaoa_xy(n)
    elif unitary_type == "clifford_only":
        components = _generate_clifford_only(n)
    else:  # pragma: no cover - normalize_unitary_type makes this unreachable.
        raise AssertionError(unitary_type)

    components = _validate_components(unitary_type, n, components)
    if unitary_type != "random_type" and _component_depth_sum(components) < 50:
        raise AssertionError(f"{unitary_type} must expose at least 50 depth slots.")
    return components


def get_unitary_components(unitary_type: str, num_qubits: int) -> tuple[ComponentSpec, ...]:
    """Return the full deterministic component sequence."""
    return _cached_unitary_components(
        normalize_unitary_type(unitary_type), int(num_qubits)
    )


def get_unitary_max_depth(unitary_type: str, num_qubits: int) -> int:
    return _component_depth_sum(get_unitary_components(unitary_type, num_qubits))


def select_unitary_components(unitary_type: str, num_qubits: int,
                              depth: int) -> tuple[SelectedComponent, ...]:
    """Select exactly ``depth`` physical-depth slots, retaining a cut macro atomically.

    DEPTH may end inside CZ, CRZ, CRX, RZZ, RXX or RYY.  The runtime
    decomposer emits exactly the retained prefix, and local folding treats that
    truncated prefix as one component G_i.
    """
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise ValueError("DEPTH must be an integer.")
    components = get_unitary_components(unitary_type, num_qubits)
    maximum = _component_depth_sum(components)
    if not 1 <= depth <= maximum:
        raise ValueError(
            f"For UNITARY_TYPE={normalize_unitary_type(unitary_type)!r} and "
            f"NUM_U_QUBITS={num_qubits}, DEPTH must be between 1 and {maximum}; "
            f"got {depth}."
        )

    selected: list[SelectedComponent] = []
    consumed = 0
    for index, component in enumerate(components):
        if consumed >= depth:
            break
        included = min(component.depth, depth - consumed)
        selected.append(SelectedComponent(
            index=index,
            label=component.label,
            full_depth=component.depth,
            included_depth=included,
            gates=component.gates,
            family=component.family,
            depth_start=consumed + 1,
            depth_stop=consumed + included,
        ))
        consumed += included
    if consumed != depth:
        raise AssertionError("Selected component depth accounting failed.")
    return tuple(selected)


def selected_active_qubits(component: SelectedComponent) -> tuple[int, ...]:
    return tuple(sorted({q for gate in component.gates for q in gate.qubits}))


def selected_unitary_summary(unitary_type: str, num_qubits: int, depth: int) -> str:
    selected = select_unitary_components(unitary_type, num_qubits, depth)
    maximum = get_unitary_max_depth(unitary_type, num_qubits)
    cut = selected[-1].included_depth < selected[-1].full_depth
    suffix = (
        f"; final component cut to {selected[-1].included_depth}/"
        f"{selected[-1].full_depth} slots" if cut else ""
    )
    return (
        f"type={normalize_unitary_type(unitary_type)}, qubits={num_qubits}, "
        f"selected depth={depth}/{maximum}, local components={len(selected)}{suffix}"
    )


# Defensive import-time checks.
_ORIGINAL_LENGTHS = {1: 18, 2: 18, 3: 18, 4: 16}
if {n: len(layers) for n, layers in U_GATES.items()} != _ORIGINAL_LENGTHS:
    raise AssertionError("The historical U_GATES dictionary lengths were changed.")
if any(RANDOM_TYPE_U_GATES[n][:len(U_GATES[n])] != U_GATES[n]
       for n in (1, 2, 3)):
    raise AssertionError("The historical 1-3q random_type prefixes were changed.")
if len(RANDOM_TYPE_U_GATES[4]) != 16:
    raise AssertionError("The fixed 4q random_type prefix must remain 16 layers.")
