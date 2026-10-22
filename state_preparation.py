# ============================================================
# Initial state preparation
# ============================================================

alpha_0 = 0.7
alpha_1 = 0.6
alpha_2 = 0.5
alpha_3 = 0.4
alpha_4 = 0.3

beta_0 = 0.4
beta_1 = 0.3
beta_2 = 0.2
beta_3 = 0.1
beta_4 = 0.0


# ============================================================
# State preparation
# ============================================================

def prepare_initial_state(qc, wires):
    """Prepare the fixed 1-5 qubit input state used by all ZNE methods.

    Existing 1-4 qubit behavior is preserved exactly:
      1. apply all local RY rotations,
      2. apply all local RZ rotations,
      3. entangle a nearest-neighbor chain with CX gates.

    The 5q extension continues the parameter pattern with alpha_4=0.3 and
    beta_4=0.0, followed by CX(3,4).
    """
    num_qubits = len(wires)

    if num_qubits == 1:
        qc.ry(alpha_0, wires[0])
        qc.rz(beta_0, wires[0])

    elif num_qubits == 2:
        qc.ry(alpha_0, wires[0])
        qc.ry(alpha_1, wires[1])
        qc.rz(beta_0, wires[0])
        qc.rz(beta_1, wires[1])
        qc.cx(wires[0], wires[1])

    elif num_qubits == 3:
        qc.ry(alpha_0, wires[0])
        qc.ry(alpha_1, wires[1])
        qc.ry(alpha_2, wires[2])
        qc.rz(beta_0, wires[0])
        qc.rz(beta_1, wires[1])
        qc.rz(beta_2, wires[2])
        qc.cx(wires[0], wires[1])
        qc.cx(wires[1], wires[2])

    elif num_qubits == 4:
        qc.ry(alpha_0, wires[0])
        qc.ry(alpha_1, wires[1])
        qc.ry(alpha_2, wires[2])
        qc.ry(alpha_3, wires[3])
        qc.rz(beta_0, wires[0])
        qc.rz(beta_1, wires[1])
        qc.rz(beta_2, wires[2])
        qc.rz(beta_3, wires[3])
        qc.cx(wires[0], wires[1])
        qc.cx(wires[1], wires[2])
        qc.cx(wires[2], wires[3])

    elif num_qubits == 5:
        qc.ry(alpha_0, wires[0])
        qc.ry(alpha_1, wires[1])
        qc.ry(alpha_2, wires[2])
        qc.ry(alpha_3, wires[3])
        qc.ry(alpha_4, wires[4])
        qc.rz(beta_0, wires[0])
        qc.rz(beta_1, wires[1])
        qc.rz(beta_2, wires[2])
        qc.rz(beta_3, wires[3])
        qc.rz(beta_4, wires[4])
        qc.cx(wires[0], wires[1])
        qc.cx(wires[1], wires[2])
        qc.cx(wires[2], wires[3])
        qc.cx(wires[3], wires[4])

    else:
        raise ValueError(
            f"Unsupported number of qubits = {num_qubits}; supported range is 1-5."
        )
