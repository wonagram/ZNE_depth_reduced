"""Local depth-reduced folding with intermediate correction and qubit reuse.

One user-declared depth-one layer is one component G_i. For each component:
prepare singlets -> all copies -> all Bell measurements -> Pauli corrections
-> reset measured wires -> next component. No live logical state is reset.

A fixed pool of n * fold_factor qubits is used. Three independent settings:
  input_placement: 'front' (the previous layouts) or 'first_use' (inputs placed
  offline according to their first component use, below prior folding wires).
  qubit_assignment: 'fixed' lanes r*n+k, or a 'dynamic' shared free pool.
  reuse_pattern: 'snake' or 'cycle'. All eight combinations are supported.
Fixed routing retains the previous per-lane behavior. Dynamic cycle starts
from the top of the free pool in every component. Dynamic snake alternates
top-to-bottom / bottom-to-top free-workspace allocation per folded component.
In first_use mode the planner grows a compact prefix as new inputs appear;
unused higher-index wires are not part of that component's workspace pool.
ALL inputs, including dormant entangled inputs, are nevertheless prepared
once at the beginning on plan['input_ids']; no late input reinitialization.
Live inputs/spectators are pinned, not silently relabelled as free qubits.
These are circuit-index layouts, not backend coupling-map optimization.
No SWAP gates or extra relocation teleportations are inserted.

Source -> classical bit 1; middle -> bit 0, even for upward links.
Corrections: 00 -> Z then X, 01 -> Z, 10 -> X, 11 -> I, up to global phase.
Classical Bell registers are not reused: each event keeps its own history.

The noiseless checker executes the actual measurement, reset and if_test
instructions, exhaustively. No component is replaced by an ideal target.
Qiskit imports are local so the allocator can be inspected with NumPy only.
"""

from numbers import Integral
import numpy as np

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_PAULIS = {"i": _I, "x": _X, "y": _Y, "z": _Z}


def _dynamic_blocks(before, active, factor, physical, pattern, direction):
    """Choose new copy blocks from an end of the shared free-wire pool.

    The first copy stays on the live input wires. All other live logical
    wires (including spectators) are excluded from workspace allocation.

    Cycle: restart at the TOP of the free pool for EVERY component.
    Snake/down: likewise scan the free pool from top to bottom.
    Snake/up: scan from the BOTTOM of the free pool toward the top.

    Top/bottom mean smaller/larger physical indices, respectively. Only the
    workspace blocks follow this order; the first copy cannot be relocated
    without an extra state transfer. Physical indices within each workspace
    block remain ascending, preserving the active-logical ordering.
    """
    first = {k: before[k] for k in active}
    if factor == 1 or not active:
        return [first] + [{} for _ in range(factor - 1)]
    if pattern not in ('snake', 'cycle'):
        raise ValueError("pattern must be 'snake' or 'cycle'.")
    if pattern == 'snake' and direction not in ('down', 'up'):
        raise ValueError("A folded snake component needs direction 'down' or 'up'.")

    width = len(active)
    needed = width * (factor - 1)
    live = set(before)  # Includes ALL live inputs and spectators.
    free = [q for q in range(physical) if q not in live]
    if len(free) < needed:
        raise AssertionError('Not enough free workspace in the fixed pool.')

    if pattern == 'snake' and direction == 'up':
        free.reverse()

    selected = free[:needed]
    groups = [sorted(selected[t:t + width])
              for t in range(0, needed, width)]
    return [first] + [dict(zip(active, group)) for group in groups]


def _clean_layers(layers, n):
    """Validate declared depth-one components without regrouping their gates."""
    result = []
    for j, layer in enumerate(layers):
        occupied, gates = set(), []
        for gate in layer:
            if not isinstance(gate, (tuple, list)) or len(gate) != 3:
                raise ValueError(f"Unsupported gate in layer {j + 1}: {gate}")
            kind = gate[0]
            if kind not in ("x", "y", "z", "cx"):
                raise ValueError(f"Unsupported gate in layer {j + 1}: {gate}")
            indices = gate[1:3] if kind == "cx" else gate[1:2]
            if any(isinstance(k, bool) or not isinstance(k, Integral)
                   or not 0 <= k < n for k in indices):
                raise ValueError(f"Invalid logical qubit in gate {gate}")
            if len(set(indices)) != len(indices):
                raise ValueError(f"A CX needs distinct control and target: {gate}")
            if occupied.intersection(indices):
                raise ValueError(
                    f"Layer {j + 1} has depth greater than one: overlapping gates. "
                    "Put gates sharing a qubit in separate declared layers."
                )
            occupied.update(indices)
            if kind == "cx":
                gates.append((kind, int(gate[1]), int(gate[2])))
            else:
                angle = float(gate[2])
                if not np.isfinite(angle):
                    raise ValueError(f"Non-finite rotation angle in {gate}")
                gates.append((kind, int(gate[1]), angle))
        result.append((gates, sorted(occupied)))
    return result


def make_local_plan(num_qubits, layers, fold_factor, *, reuse_pattern="snake",
                    qubit_assignment="fixed", input_placement="front"):
    """Plan all 2 x 2 x 2 placement/assignment/routing combinations.

    input_placement='front': EXACT previous four layouts; input k is on q[k].
    input_placement='first_use': plan input positions offline, in order of
    first use by a component. A newly encountered logical input gets the next
    index BELOW all previously allocated folding wires, never a recycled wire.
    Inputs first used together are placed together in ascending logical order.

    First-use + fixed: allocate permanent lanes for each newly encountered
    group, interleaved within that group (width = group size, not final n).
    First-use + dynamic: extend a compact working prefix only when needed;
    all free wires in that prefix are shared. Snake/up takes bottom free wires,
    snake/down and cycle take top free wires. Dormant inputs never enter it.

    The complete n-qubit input is STILL PREPARED ONCE AT THE START on input_ids.
    'First use' specifies SPATIAL placement, not delayed state initialization.
    This preserves arbitrary input entanglement and never borrows a dormant
    input wire. The complete physical pool remains n*fold_factor qubits.
    """
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, Integral):
        raise ValueError("num_qubits must be an integer.")
    if not 1 <= num_qubits <= 4:
        raise ValueError("This implementation supports 1-4 logical qubits.")
    if (isinstance(fold_factor, bool) or not isinstance(fold_factor, Integral)
            or fold_factor < 1 or fold_factor % 2 != 1):
        raise ValueError("fold_factor must be a positive odd integer.")
    if reuse_pattern not in ("snake", "cycle"):
        raise ValueError("reuse_pattern must be 'snake' or 'cycle'.")
    if qubit_assignment not in ("fixed", "dynamic"):
        raise ValueError("qubit_assignment must be 'fixed' or 'dynamic'.")
    if input_placement not in ("front", "first_use"):
        raise ValueError("input_placement must be 'front' or 'first_use'.")

    n, factor = int(num_qubits), int(fold_factor)
    folds, physical = (factor - 1) // 2, n * factor
    validated_layers = _clean_layers(layers, n)
    front = input_placement == 'front'
    input_ids = list(range(n)) if front else [None] * n
    current_wire = input_ids.copy()
    rows = [0] * n
    lanes = ({k: [r*n+k for r in range(factor)] for k in range(n)}
             if front and qubit_assignment == 'fixed' else {})
    frontier = physical if front else 0  # Exclusive layout prefix end.
    first_use = [None] * n
    seen = set()
    stages, events, resource_pairs = [], [], []
    dynamic_component_count = 0

    def place_inputs(logicals):
        # This chooses an INITIAL layout offline; it does not add a runtime
        # initialize operation at a later component.
        nonlocal frontier
        if front or not logicals:
            return
        start, width = frontier, len(logicals)
        for i, k in enumerate(logicals):
            input_ids[k] = current_wire[k] = start + i
            if qubit_assignment == 'fixed':
                lanes[k] = [start + r*width + i for r in range(factor)]
        frontier += width * (factor if qubit_assignment == 'fixed' else 1)
        if frontier > physical:
            raise AssertionError('Input placement exceeded n*fold_factor.')

    for j, (gates, active) in enumerate(validated_layers):
        newly_used = [k for k in active if k not in seen]
        place_inputs(newly_used)
        for k in newly_used:
            first_use[k] = j
        seen.update(newly_used)
        before = current_wire.copy()
        directions = {}

        if qubit_assignment == 'fixed':
            paths = {}
            for k in active:
                if factor == 1:
                    path, direction = [0], 'none'
                elif reuse_pattern == 'snake':
                    if rows[k] not in (0, factor - 1):
                        raise AssertionError('A snake lane must start at an endpoint.')
                    step = 1 if rows[k] == 0 else -1
                    path = [rows[k] + step*t for t in range(factor)]
                    direction = 'down' if step == 1 else 'up'
                else:
                    path = [(rows[k] + t) % factor for t in range(factor)]
                    direction = 'down/wrap'
                paths[k], directions[k] = path, direction
            blocks = [{k: lanes[k][paths[k][t]] for k in active}
                      for t in range(factor)]
            for k in active:
                rows[k] = paths[k][-1]
            routing_hint = 'per-lane'
        else:
            if not front:
                # With m introduced live inputs, a component of width a needs
                # m + a*(factor-1) places. Do not let an upward scan jump to
                # the unused bottom of the final n*factor pool.
                needed = len(active) * (factor - 1)
                frontier = max(frontier, len(seen) + needed)
            if factor == 1 or not active:
                routing_hint = 'none'
            elif reuse_pattern == 'cycle':
                routing_hint = 'down'
            else:
                routing_hint = 'down' if dynamic_component_count % 2 == 0 else 'up'
            blocks = _dynamic_blocks(before, active, factor, frontier,
                                     reuse_pattern, routing_hint)
            directions = {k: routing_hint for k in active}
            if active and factor > 1:
                dynamic_component_count += 1

        stage_events = []
        for fold in range(folds):
            source, middle, destination = blocks[2*fold:2*fold+3]
            for k in active:
                stage_events.append(len(events))
                events.append({
                    'layer': j, 'fold': fold, 'logical': k,
                    'source': source[k], 'middle': middle[k],
                    'destination': destination[k],
                })
                resource_pairs.append((middle[k], destination[k]))
        workspace = [wire for block in blocks[1:] for wire in block.values()]
        for k in active:
            current_wire[k] = blocks[-1][k]
        measured = [wire for i in stage_events
                    for wire in (events[i]['source'], events[i]['middle'])]
        stages.append({
            'inputs_before_layer': before, 'gates': gates, 'active': active,
            'blocks': blocks, 'events': stage_events,
            'outputs_after_layer': current_wire.copy(),
            'workspace': workspace, 'reset_ids': measured,
            'directions': directions, 'routing_hint': routing_hint,
            'physical_paths': {k: [block[k] for block in blocks] for k in active},
            'newly_used_inputs': {k: input_ids[k] for k in newly_used},
            'introduced_logicals': sorted(seen), 'layout_limit': frontier,
        })

    # Even an input that never appears in the selected layers is part of phi
    # and of the final output. Reserve it below the folding prefix as well.
    never_used = [k for k in range(n) if k not in seen]
    place_inputs(never_used)
    for stage in stages:
        for key in ('inputs_before_layer', 'outputs_after_layer'):
            stage[key] = [input_ids[k] if wire is None else wire
                          for k, wire in enumerate(stage[key])]
    current_wire = [input_ids[k] if wire is None else wire
                    for k, wire in enumerate(current_wire)]
    plan = {
        'n': n, 'fold_factor': factor, 'num_physical_qubits': physical,
        'reuse_pattern': reuse_pattern, 'qubit_assignment': qubit_assignment,
        'input_placement': input_placement, 'input_ids': input_ids,
        'first_use_layers': first_use, 'fixed_lanes': lanes,
        'layout_used': frontier,
        'stages': stages, 'events': events, 'resource_pairs': resource_pairs,
        'output_ids': current_wire.copy(),
    }
    _validate_local_plan(plan)
    return plan


def _validate_local_plan(plan):
    """Audit physical liveness, input placement and reuse independently."""
    n, factor, physical = plan['n'], plan['fold_factor'], plan['num_physical_qubits']
    if physical != n * factor:
        raise ValueError('Physical pool must have n * fold_factor wires.')
    input_ids = plan.get('input_ids')
    placement = plan.get('input_placement')
    if placement not in ('front', 'first_use') or input_ids is None:
        raise ValueError('Rebuild this plan using the input-placement helper.')
    if (len(input_ids) != n or len(set(input_ids)) != n
            or any(not isinstance(q, Integral) or isinstance(q, bool)
                   or not 0 <= q < physical for q in input_ids)):
        raise ValueError('Input layout must have one distinct physical wire per logical input.')
    if placement == 'front' and input_ids != list(range(n)):
        raise ValueError('Front placement must use the first n physical qubits.')
    live = input_ids.copy()   # ALL n inputs exist, even before their first G_i.
    free = set(range(physical)) - set(live)
    seen, expected_first_use = set(), [None] * n
    frontier = physical if placement == 'front' else 0
    next_event, pairs = 0, []
    for j, stage in enumerate(plan['stages']):
        active, blocks = stage['active'], stage['blocks']
        _, expected_active = _clean_layers([stage['gates']], n)[0]
        if active != expected_active:
            raise ValueError(f'G{j+1}: active logical qubits do not match its gates.')
        new = [k for k in active if k not in seen]
        if stage['newly_used_inputs'] != {k: input_ids[k] for k in new}:
            raise ValueError(f'G{j+1}: incorrect first-use input record.')
        if placement == 'first_use':
            if [input_ids[k] for k in new] != list(range(frontier, frontier + len(new))):
                raise ValueError(f'G{j+1}: a new input was not placed below the previous prefix.')
            frontier += len(new) * (factor if plan['qubit_assignment'] == 'fixed' else 1)
        for k in new:
            expected_first_use[k] = j
        seen.update(new)
        if placement == 'first_use' and plan['qubit_assignment'] == 'dynamic':
            frontier = max(frontier, len(seen) + len(active)*(factor-1))
        if stage['layout_limit'] != frontier or frontier > physical:
            raise ValueError(f'G{j+1}: inconsistent compact layout prefix.')
        if stage['introduced_logicals'] != sorted(seen):
            raise ValueError(f'G{j+1}: incorrect introduced-input set.')
        if stage['inputs_before_layer'] != live:
            raise ValueError(f'G{j+1}: input map lost a live logical state.')
        if len(blocks) != factor or any(set(b) != set(active) for b in blocks):
            raise ValueError(f'G{j+1}: invalid copy blocks.')
        if blocks[0] != {k: live[k] for k in active}:
            raise ValueError(f'G{j+1}: first copy is not on the live input wires.')
        used = [b[k] for b in blocks for k in active]
        if len(used) != len(set(used)) or any(not 0 <= q < frontier for q in used):
            raise ValueError(f'G{j+1}: overlapping or out-of-prefix physical copies.')
        workspace = [b[k] for b in blocks[1:] for k in active]
        if stage['workspace'] != workspace or not set(workspace) <= free:
            raise ValueError(f'G{j+1}: workspace overwrites live data or a reserved future input.')
        if plan['qubit_assignment'] == 'fixed':
            for k in active:
                lane = plan['fixed_lanes'][k]
                if len(lane) != factor or set(b[k] for b in blocks) != set(lane):
                    raise ValueError(f'G{j+1}: fixed assignment left its permanent lane.')
        expected_events, measured = [], []
        for fold in range((factor - 1) // 2):
            for k in active:
                e = {
                    'layer': j, 'fold': fold, 'logical': k,
                    'source': blocks[2*fold][k], 'middle': blocks[2*fold+1][k],
                    'destination': blocks[2*fold+2][k],
                }
                if next_event >= len(plan['events']) or plan['events'][next_event] != e:
                    raise ValueError(f'G{j+1}: Bell links disagree with the copy maps.')
                expected_events.append(next_event)
                next_event += 1
                measured.extend((e['source'], e['middle']))
                pairs.append((e['middle'], e['destination']))
        if stage['events'] != expected_events or stage['reset_ids'] != measured:
            raise ValueError(f'G{j+1}: measurements/resets do not match the Bell links.')
        after = live.copy()
        for k in active:
            after[k] = blocks[-1][k]
        if (stage['outputs_after_layer'] != after or len(set(after)) != n
                or set(measured).intersection(after)):
            raise ValueError(f'G{j+1}: invalid outputs or reset of a live state.')
        free = (free - set(workspace)) | set(measured)
        live = after
        if free != set(range(physical)) - set(live):
            raise ValueError(f'G{j+1}: physical-wire accounting failed.')
    never_used = [k for k in range(n) if k not in seen]
    if placement == 'first_use':
        if [input_ids[k] for k in never_used] != list(range(frontier, frontier + len(never_used))):
            raise ValueError('Never-used inputs must be reserved below the folding prefix.')
        frontier += len(never_used) * (factor if plan['qubit_assignment'] == 'fixed' else 1)
    if frontier != plan['layout_used'] or frontier > physical:
        raise ValueError('Final layout width is inconsistent.')
    if expected_first_use != plan['first_use_layers']:
        raise ValueError('First-use layers are inconsistent with the logical circuit.')
    if (next_event != len(plan['events']) or pairs != plan['resource_pairs']
            or live != plan['output_ids']):
        raise ValueError('Final map or resource history is inconsistent.')


def _single_on_n(matrix, logical, n):
    result = np.array([[1]], dtype=complex)
    for k in reversed(range(n)):
        result = np.kron(result, matrix if k == logical else _I)
    return result

def _layer_matrix(gates, n):
    dim = 1 << n
    result = np.eye(dim, dtype=complex)
    for kind, first, last in gates:
        if kind == "cx":
            indices = np.arange(dim)
            destinations = indices ^ (((indices >> first) & 1) << last)
            matrix = np.zeros((dim, dim), dtype=complex)
            matrix[destinations, indices] = 1
        else:
            rotation = (np.cos(last / 2) * _I -
                        1j * np.sin(last / 2) * _PAULIS[kind])
            matrix = _single_on_n(rotation, first, n)
        result = matrix @ result
    return result

def _register_name(event):
    return f'c{event["layer"]}_{event["fold"]}_{event["logical"]}'

def _default_correction(qc, register, target):
    """The same branch convention as the supplied experiment script."""
    with qc.if_test((register, 0)):
        qc.z(target)
        qc.x(target)
    with qc.if_test((register, 1)):
        qc.z(target)
    with qc.if_test((register, 2)):
        qc.x(target)

def build_intermediate_local_circuit(
    plan, *, prepare_initial_state, apply_rotation, apply_noise,
    psi_minus, bell_measure, apply_correction=None, use_noise=False,
    stage_barriers=True,
):
    """Build the full dynamic circuit, including resets and final readout.

    All Bell measurements in ONE component precede its corrections. For
    factor 5 both links correct the same component output, latest link first.
    The following component starts on those corrected, un-reset live wires.

    Keep the supplied noise policy: explicit channels are inserted only after
    copied rotation/CX gates, NOT after auxiliary gates, measurement or reset.
    Reset imperfections/idle noise therefore require a separate noise model.
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    _validate_local_plan(plan)
    q = QuantumRegister(plan['num_physical_qubits'], 'q')
    qc = QuantumCircuit(q)
    classical_pairs = []
    for event in plan['events']:
        register = ClassicalRegister(2, _register_name(event))
        qc.add_register(register)
        classical_pairs.append(register)
    correction = apply_correction or _default_correction
    prepare_initial_state(qc, [q[k] for k in plan['input_ids']])
    prep_end = len(qc.data)
    stage_records = []

    for j, stage in enumerate(plan['stages']):
        start = len(qc.data)
        logical_scope = (range(plan['n']) if plan['input_placement'] == 'front'
                         else stage['introduced_logicals'])
        stage_ids = {stage['inputs_before_layer'][k] for k in logical_scope}
        stage_ids.update(stage['workspace'])
        wires = [q[k] for k in sorted(stage_ids)]
        if stage_barriers and wires:
            qc.barrier(*wires, label=f'G{j + 1} start')

        # 1. Workspace is either initially |0> or reset after an earlier G_i.
        for i in stage['events']:
            e = plan['events'][i]
            psi_minus(qc, [q[e['middle']], q[e['destination']]])
        if stage_barriers and stage['events']:
            qc.barrier(*wires, label='|\u03c9\u27e9 prep')

        # 2. Copy parity is LOGICAL copy parity, not physical wire number.
        #    This is essential when the module travels upward.
        for block_index, block in enumerate(stage['blocks']):
            dressed = block_index % 2 == 1
            for kind, first, last in stage['gates']:
                if kind in ('x', 'y', 'z'):
                    target = q[block[first]]
                    apply_rotation(qc, target, kind, last)
                    if use_noise:
                        apply_noise(qc, [target], kind)
                else:
                    control, target = q[block[first]], q[block[last]]
                    if dressed:
                        qc.y(control)
                        qc.y(target)
                    qc.cx(control, target)
                    if use_noise:
                        apply_noise(qc, [control, target], 'cx')
                    if dressed:
                        qc.y(control)
                        qc.y(target)

        # 3. Complete all Bell-basis changes before starting measurement.
        if stage_barriers and stage['events']:
            qc.barrier(*wires, label=f'G{j + 1} foldings')
        for i in stage['events']:
            e = plan['events'][i]
            bell_measure(qc, [q[e['source']], q[e['middle']]])

        # 4. Displayed register c[1]c[0] ALWAYS means source,middle.
        #    Do not sort by physical number when travelling upward.
        for i in stage['events']:
            e = plan['events'][i]
            qc.measure(q[e['source']], classical_pairs[i][1])
            qc.measure(q[e['middle']], classical_pairs[i][0])

        # 5. Correct all links on the LAST output of THIS component.
        for i in reversed(stage['events']):
            e = plan['events'][i]
            target = stage['outputs_after_layer'][e['logical']]
            correction(qc, classical_pairs[i], q[target])
        corrections_end = len(qc.data)

        # 6. Reclaim measured wires ONLY. Reset does not erase classical bits.
        #    Never reset any output or inactive spectator carrying live data.
        for physical in stage['reset_ids']:
            qc.reset(q[physical])

        stage_records.append({
            'layer': j, 'start': start, 'end': len(qc.data),
            'corrections_end': corrections_end,
            'output_ids': stage['outputs_after_layer'].copy(),
            'reset_ids': stage['reset_ids'].copy(),
        })
        # No extra 'Bell basis' or 'corrected' barriers; next G starts here.

    output_start = len(qc.data)
    c_out = ClassicalRegister(plan['n'], 'out')
    qc.add_register(c_out)
    for k, physical in enumerate(plan['output_ids']):
        qc.measure(q[physical], c_out[k])
    qc.metadata = dict(qc.metadata or {})
    qc.metadata['intermediate_local_folding'] = {
        'version': 4, 'qubit_reuse': True,
        'input_placement': plan['input_placement'], 'input_ids': plan['input_ids'].copy(),
        'qubit_assignment': plan['qubit_assignment'],
        'prep_end': prep_end, 'stages': stage_records,
        'output_start': output_start, 'output_ids': plan['output_ids'].copy(),
        'use_noise': bool(use_noise), 'reuse_pattern': plan['reuse_pattern'],
    }
    return qc


def describe_local_plan(plan):
    m = len(plan['events'])
    print(f"Logical qubits: {plan['n']}; physical qubits: "
          f"{plan['num_physical_qubits']}; fold factor: {plan['fold_factor']}")
    print(f"Reusable pool; assignment={plan['qubit_assignment']}; "
          f"routing={plan['reuse_pattern']}; input placement={plan['input_placement']}")
    print('Initial input map (logical -> physical):', dict(enumerate(plan['input_ids'])))
    print('First use of each input (1-based G index, None = never):',
          [None if j is None else j + 1 for j in plan['first_use_layers']])
    print(f"Bell links: {m}; possible full Bell histories: {4 ** m}")
    print('Output map (logical -> physical):', dict(enumerate(plan['output_ids'])))
    for j, stage in enumerate(plan['stages']):
        print(f"  G{j + 1}: active={stage['active']}, "
              f"directions={stage['directions']}, "
              f"corrected outputs={stage['outputs_after_layer']}, "
              f"reset={stage['reset_ids']}")
        print('    first-used inputs:', stage['newly_used_inputs'],
              '; working prefix:', stage['layout_limit'])
        print('    copies in active-logical order:',
              [[block[k] for k in stage['active']] for block in stage['blocks']])
        if plan['qubit_assignment'] == 'dynamic':
            spectators = {k: q for k, q in enumerate(stage['inputs_before_layer'])
                          if k not in stage['active']}
            print('    protected spectators (logical -> physical):', spectators)


def _apply_matrix(vector, matrix, positions):
    """Apply a k-qubit matrix; positions[0] is its least-significant bit."""
    n = vector.size.bit_length() - 1
    if not positions:
        return vector * complex(matrix[0, 0])
    positions = list(positions)
    rest = [k for k in range(n) if k not in positions]
    axes = positions + rest
    tensor = vector.reshape([2] * n, order='F').transpose(axes)
    result = matrix @ tensor.reshape((1 << len(positions), -1), order='F')
    result = result.reshape([2] * n, order='F')
    result = result.transpose(np.argsort(axes))
    return result.reshape(-1, order='F')

def _project_and_remove(vector, position, outcome):
    """Project one measured qubit, then discard its now separable axis."""
    n = vector.size.bit_length() - 1
    projected = np.take(
        vector.reshape([2] * n, order='F'), outcome, axis=position
    ).reshape(-1, order='F')
    probability = float(np.vdot(projected, projected).real)
    if probability <= 0.0:
        return 0.0, None
    return probability, projected / np.sqrt(probability)

def _logical_vector(vector, live_ids, output_ids, atol):
    """Extract ordered logical outputs; reset/unused workspace must be |0>."""
    positions = [live_ids.index(k) for k in output_ids]
    logical_indices = np.arange(1 << len(output_ids), dtype=np.int64)
    indices = np.zeros(logical_indices.size, dtype=np.int64)
    for logical, physical_position in enumerate(positions):
        indices |= ((logical_indices >> logical) & 1) << physical_position
    result = vector[indices].copy()
    norm = float(np.vdot(result, result).real)
    if abs(norm - 1.0) > max(100 * atol, 1e-8):
        raise ValueError(
            'At a component checkpoint, reset/unused workspace are not |0>. '
            'The checker expects the circuit produced by '
            'build_intermediate_local_circuit without later rewrites.'
        )
    if norm == 0.0:
        raise ValueError('Zero-norm logical state at a checkpoint.')
    return result / np.sqrt(norm)

def _compile_unitary_records(circuit, qmap, cmap, inherited=()):
    """Read ACTUAL Qiskit operations/if_test bodies into a numerical tape.

    Conditions supported: (Clbit, int) or (ClassicalRegister, int), which
    includes the correction function supplied in the user's main file.
    No stochastic noise, loops, or conditional measurements are
    silently ignored. Unsupported operations raise an informative error.
    """
    from qiskit import ClassicalRegister
    from qiskit.quantum_info import Operator

    records = []
    phase = float(circuit.global_phase)
    if phase:
        records.append(('unitary', np.array([[np.exp(1j * phase)]]), (), inherited))
    for item in circuit.data:
        op = item.operation
        name = op.name
        qids = tuple(qmap[circuit.find_bit(q).index] for q in item.qubits)
        cids = tuple(cmap[circuit.find_bit(c).index] for c in item.clbits)
        if name == 'barrier':
            continue
        if name == 'measure':
            if inherited or len(qids) != 1 or len(cids) != 1:
                raise ValueError('The exact checker supports unconditional 1-qubit measurements.')
            records.append(('measure', qids[0], cids[0]))
            continue
        if name == 'reset':
            if inherited or len(qids) != 1:
                raise ValueError('Checker supports unconditional 1-qubit reset only.')
            records.append(('reset', qids[0]))
            continue
        if name == 'if_else':
            condition = op.condition
            if not isinstance(condition, tuple) or len(condition) != 2:
                raise ValueError('Checker needs a (Clbit/register, integer) if_test condition.')
            control, value = condition
            if isinstance(control, ClassicalRegister):
                bits = tuple(cmap[circuit.find_bit(b).index] for b in control)
            else:
                bits = (cmap[circuit.find_bit(control).index],)
            for branch_index, body in enumerate(op.blocks):
                if body is None:
                    continue
                guards = inherited + ((bits, int(value), branch_index == 0),)
                records.extend(_compile_unitary_records(body, qids, cids, guards))
            continue
        if name in ('initialize', 'kraus', 'superop', 'quantum_channel',
                    'for_loop', 'while_loop', 'switch_case', 'store'):
            raise ValueError(
                f'Unsupported instruction {name!r} after input preparation. '
                'Run the checker without noise or unsupported circuit rewrites.'
            )
        if getattr(op, 'condition', None) is not None:
            raise ValueError('Use if_test rather than a legacy conditioned gate in the checker.')
        try:
            matrix = np.asarray(Operator(op).data, dtype=complex)
        except Exception as exc:
            raise ValueError(f'Cannot interpret {name!r} as a unitary gate.') from exc
        dim = 1 << len(qids)
        if (matrix.shape != (dim, dim) or
                not np.allclose(matrix.conj().T @ matrix, np.eye(dim), atol=1e-9, rtol=0)):
            raise ValueError(f'Instruction {name!r} is not a unitary of the expected size.')
        records.append(('unitary', matrix, qids, inherited))
    return records

def _condition_holds(memory, guards):
    for bits, expected, sense in guards:
        value = sum(int(memory[bit]) << k for k, bit in enumerate(bits))
        if (value == expected) != sense:
            return False
    return True

def check_intermediate_local_folding(
    qc, plan, *, prepare_initial_state, max_physical_qubits=20,
    max_branches=4096, print_each=True, atol=1e-9,
):
    """Exhaustive noiseless check of the ACTUAL dynamic circuit before readout.

    At each mid-circuit measurement a saved premeasurement branch vector is
    projected onto both possible outcomes. The classical memory is updated,
    the matching if_test bodies are applied by matrix multiplication, and
    THAT corrected vector (not an ideal substitute) enters the next stage.

    Measured qubits are temporarily factored out of the branch tensor. Their
    actual computational-basis values are retained. Reset changes that value
    to zero, and the next gate on a reused wire re-inserts its basis factor.
    This is exact conditioning, not an assumed ideal component replacement. Final output measurements are checked for correct wiring
    but are not applied, so the logical output state can be compared with U|phi>.

    Returns aggregate statistics; no backend job is submitted. Global phase
    is reported but ignored in fidelity. Both intermediate and final states
    are tested, so an intermediate error cannot be hidden by a later fix.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    _validate_local_plan(plan)
    if atol <= 0 or not np.isfinite(atol):
        raise ValueError('atol must be positive and finite.')
    if qc.num_qubits != plan['num_physical_qubits']:
        raise ValueError('Circuit/plan physical-qubit counts differ.')
    n, physical = plan['n'], qc.num_qubits
    expected_branches = 4 ** len(plan['events'])
    if physical > max_physical_qubits:
        raise ValueError(
            f'Exact check needs {physical} physical qubits; limit is '
            f'{max_physical_qubits}. Reduce NUM_U_QUBITS/fold factor or deliberately '
            'raise CHECK_MAX_PHYSICAL_QUBITS after checking memory.'
        )
    if expected_branches > max_branches:
        raise ValueError(
            f'Exact check has {expected_branches} Bell histories; limit is '
            f'{max_branches}. Reduce DEPTH/fold factor or deliberately '
            'raise CHECK_MAX_BRANCHES. No sampling has been substituted.'
        )
    import sys
    if 2 * len(plan['events']) + 100 >= sys.getrecursionlimit():
        raise ValueError('Too many measurement levels for exhaustive recursive checking. '
                         'Reduce DEPTH/fold factor. No sampling has been substituted.')
    meta = (qc.metadata or {}).get('intermediate_local_folding')
    if not meta or meta.get('version') != 4 or not meta.get('qubit_reuse'):
        raise ValueError('Use the untranspiled circuit returned by the intermediate builder.')
    if (meta.get('qubit_assignment') != plan['qubit_assignment']
            or meta.get('reuse_pattern') != plan['reuse_pattern']
            or meta.get('input_placement') != plan['input_placement']
            or meta.get('input_ids') != plan['input_ids']):
        raise ValueError('Circuit allocation/routing metadata differs from the plan.')
    if meta['use_noise']:
        raise ValueError('Build with use_noise=False for an exact statevector check.')
    if len(meta['stages']) != len(plan['stages']):
        raise ValueError('Circuit metadata and component plan do not match.')

    # Independent reference: the user's n-qubit input preparation, followed
    # by matrices of the ORIGINAL logical layers (not copies or corrections).
    reference_prep = QuantumCircuit(n)
    prepare_initial_state(reference_prep, list(reference_prep.qubits))
    initial = np.asarray(Statevector.from_instruction(reference_prep).data, dtype=complex)
    initial = initial / np.linalg.norm(initial)
    ideal_targets = []
    target = initial.copy()
    total_u = np.eye(1 << n, dtype=complex)
    for stage in plan['stages']:
        layer_u = _layer_matrix(stage['gates'], n)
        total_u = layer_u @ total_u
        target = layer_u @ target
        ideal_targets.append(target.copy())

    # Actual input prefix, including any pure-state initialize instruction.
    # It is evaluated separately; reference states are never fed into the
    # physical circuit's subsequent branch evolution.
    actual_prep = QuantumCircuit(physical)
    actual_prep.global_phase = qc.global_phase
    for item in qc.data[:meta['prep_end']]:
        if item.clbits:
            raise ValueError('Input preparation must not use classical bits.')
        actual_prep.append(
            item.operation, [qc.find_bit(q).index for q in item.qubits], []
        )
    vector = np.asarray(Statevector.from_instruction(actual_prep).data, dtype=complex)
    vector = vector / np.linalg.norm(vector)
    # Audit the actual preparation on the NONCONSECUTIVE initial wire map.
    # Never replace it by the reference vector, even after this check passes.
    actual_input = _logical_vector(vector, tuple(range(physical)), plan['input_ids'], atol)
    initial_fidelity = float(abs(np.vdot(initial, actual_input)) ** 2)
    if abs(1.0 - initial_fidelity) > atol:
        raise ValueError('Actual initial state differs from the logical reference. '
                         'prepare_initial_state must act on the supplied wires, '
                         'not hard-coded q[0], q[1], ... .')

    # Compile each actual component, including its actual measurement bits
    # and condition bodies. Checkpoints occur AFTER its corrections.
    tape = []
    previous_end = meta['prep_end']
    for j, record in enumerate(meta['stages']):
        if record['start'] != previous_end or record['layer'] != j:
            raise ValueError('Component ranges are inconsistent; rebuild the circuit.')
        segment = QuantumCircuit(*qc.qregs, *qc.cregs)
        for item in qc.data[record['start']:record['end']]:
            segment.append(item.operation, list(item.qubits), list(item.clbits))
        component_tape = _compile_unitary_records(
            segment, tuple(range(physical)), tuple(range(qc.num_clbits))
        )
        actual_resets = [op[1] for op in component_tape if op[0] == 'reset']
        if actual_resets != plan['stages'][j]['reset_ids']:
            raise ValueError(f"G{j + 1}: reset wires/order differs from the plan. "
                             "Rebuild the circuit; do not omit workspace resets.")
        if (record['output_ids'] != plan['stages'][j]['outputs_after_layer']
                or record['reset_ids'] != plan['stages'][j]['reset_ids']):
            raise ValueError('Checkpoint metadata does not match the reuse plan.')
        tape.extend(component_tape)
        tape.append(('checkpoint', j, tuple(record['output_ids'])))
        previous_end = record['end']
    if previous_end != meta['output_start']:
        raise ValueError('Unexpected operations outside the component ranges.')

    # Final measurement should be exactly logical k -> out[k]. Do not
    # collapse it during checking, since the target is a quantum state.
    out_registers = [r for r in qc.cregs if r.name == 'out']
    if len(out_registers) != 1:
        raise ValueError('Expected one final register named out.')
    out = out_registers[0]
    readout = list(qc.data[meta['output_start']:])
    if len(out) != n or len(readout) != n:
        raise ValueError('Unexpected final readout size.')
    for k, item in enumerate(readout):
        if (item.operation.name != 'measure' or
                qc.find_bit(item.qubits[0]).index != plan['output_ids'][k] or
                item.clbits[0] != out[k]):
            raise ValueError('Final readout is not in logical output order.')

    # Verify that the Bell bit convention, not just the number of bits,
    # agrees with the plan. Wrong wiring must not be silently accepted.
    registers = {r.name: r for r in qc.cregs}
    event_register_bits = []
    expected_measurements = []
    for event in plan['events']:
        reg = registers.get(_register_name(event))
        if reg is None or len(reg) != 2:
            raise ValueError('Missing 2-bit Bell register.')
        bits = (qc.find_bit(reg[0]).index, qc.find_bit(reg[1]).index)
        event_register_bits.append(bits)
        expected_measurements.extend([
            ('measure', event['source'], bits[1]),
            ('measure', event['middle'], bits[0]),
        ])
    actual_measurements = [rec for rec in tape if rec[0] == 'measure']
    if actual_measurements != expected_measurements:
        raise ValueError('Bell measurement wire/bit order differs from the plan.')

    report = {
        'expected_branches': expected_branches,
        'checked_branches': 0, 'passed_branches': 0,
        'probability_sum': 0.0, 'minimum_fidelity': 1.0,
        'component_counts': [0] * len(plan['stages']),
        'component_passed': [0] * len(plan['stages']),
        'component_min_fidelity': [1.0] * len(plan['stages']),
        'target': target.copy(), 'U': total_u,
        'initial_fidelity': initial_fidelity, 'input_ids': plan['input_ids'].copy(),
    }
    print('\nExact INTERMEDIATE-measurement + RESET/REUSE check (no shots).')
    print('Vector basis: |logical_(n-1) ... logical_0>.')
    print('Initial input map:', plan['input_ids'],
          f'; initial fidelity={initial_fidelity:.12f}')
    print('Target U|phi> =', np.array2string(target, precision=8, suppress_small=True))

    def materialize_wires(state, live_ids, known, qids):
        # A factored measured wire still exists physically. Reinsert its
        # ACTUAL known value, not zero unless a reset really set it to zero.
        for qid in qids:
            if qid not in live_ids:
                if qid not in known:
                    raise ValueError('A wire is neither quantum-active nor known.')
                zero = np.zeros_like(state)
                state = (np.concatenate((state, zero)) if known[qid] == 0
                         else np.concatenate((zero, state)))
                known = known.copy()
                del known[qid]
                live_ids = live_ids + (qid,)
        return state, live_ids, known

    def ordered_logical(state, live_ids, known, outputs):
        state, live_ids, known = materialize_wires(state, live_ids, known, outputs)
        if any(value != 0 for qid, value in known.items() if qid not in outputs):
            raise ValueError('Measured workspace is not reset to zero at a checkpoint.')
        return _logical_vector(state, live_ids, outputs, atol)

    def visit(pc, state, live_ids, known, memory, branch_probability):
        while pc < len(tape):
            record = tape[pc]
            kind = record[0]
            if kind == 'unitary':
                _, matrix, qids, guards = record
                if _condition_holds(memory, guards):
                    state, live_ids, known = materialize_wires(
                        state, live_ids, known, qids)
                    state = _apply_matrix(state, matrix, [live_ids.index(q) for q in qids])
            elif kind == 'measure':
                _, physical_id, bit_id = record
                state, live_ids, known = materialize_wires(
                    state, live_ids, known, (physical_id,))
                position = live_ids.index(physical_id)
                remaining = live_ids[:position] + live_ids[position + 1:]
                # state is the SAVED PREMEASUREMENT state for BOTH outcomes.
                for outcome in (0, 1):
                    probability, projected = _project_and_remove(state, position, outcome)
                    if projected is None:
                        continue
                    new_memory = memory.copy()
                    new_memory[bit_id] = outcome
                    new_known = known.copy()
                    new_known[physical_id] = outcome
                    visit(pc + 1, projected, remaining, new_known, new_memory,
                          branch_probability * probability)
                return
            elif kind == 'reset':
                qid = record[1]
                if qid not in known:
                    raise ValueError(
                        'Reset targets a quantum-active wire. This checker only '
                        'accepts reset of a measured, factored computational-basis '
                        'wire. A live logical output must not be reset.'
                    )
                known = known.copy()
                known[qid] = 0
                # No classical memory is changed by reset.
            elif kind == 'checkpoint':
                _, j, outputs = record
                logical = ordered_logical(state, live_ids, known, outputs)
                fidelity = float(abs(np.vdot(ideal_targets[j], logical)) ** 2)
                good = abs(1.0 - fidelity) <= atol
                report['component_counts'][j] += 1
                report['component_passed'][j] += int(good)
                report['component_min_fidelity'][j] = min(
                    report['component_min_fidelity'][j], fidelity
                )
            else:
                raise AssertionError(f'Unexpected tape record {kind!r}.')
            pc += 1

        logical = ordered_logical(state, live_ids, known, plan['output_ids'])
        overlap = np.vdot(target, logical)
        fidelity = float(abs(overlap) ** 2)
        good = abs(1.0 - fidelity) <= atol
        report['checked_branches'] += 1
        report['passed_branches'] += int(good)
        report['probability_sum'] += branch_probability
        report['minimum_fidelity'] = min(report['minimum_fidelity'], fidelity)
        if print_each:
            history = ' '.join(
                f'{_register_name(event)}={int(memory[bits[0]]) + 2 * int(memory[bits[1]]):02b}'
                for event, bits in zip(plan['events'], event_register_bits)
            ) or '(no Bell measurements)'
            phase = float(np.angle(overlap))
            aligned = logical * np.exp(-1j * phase)
            print(f'  {history}: p={branch_probability:.10g}, '
                  f'F={fidelity:.12f}, phase={phase:+.8f}, '
                  f'{"PASS" if good else "FAIL"}')
            print('    corrected =', np.array2string(logical, precision=8, suppress_small=True))
            print('    aligned   =', np.array2string(aligned, precision=8, suppress_small=True))

    visit(0, vector, tuple(range(physical)), {},
          np.zeros(qc.num_clbits, dtype=np.uint8), 1.0)
    print('\nComponent checkpoints (all nonzero-probability histories to that point):')
    for j, (passed, count, minimum) in enumerate(zip(
        report['component_passed'], report['component_counts'], report['component_min_fidelity']
    )):
        print(f'  G{j + 1}: {passed}/{count}; minimum fidelity={minimum:.12f}')
    print(f"Final corrected branches: {report['passed_branches']}/{report['checked_branches']}")
    print(f"Nonzero histories / possible histories: {report['checked_branches']}/{expected_branches}")
    print(f"Sum of branch probabilities: {report['probability_sum']:.12f}")
    print(f"Minimum final fidelity: {report['minimum_fidelity']:.12f}")
    report['passed'] = (
        report['checked_branches'] > 0
        and report['passed_branches'] == report['checked_branches']
        and report['component_passed'] == report['component_counts']
        and abs(report['probability_sum'] - 1.0) <= max(100 * atol, 1e-8)
    )
    print('Overall:', 'PASS' if report['passed'] else 'FAIL')
    return report

