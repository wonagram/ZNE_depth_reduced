# ============================================================
# Initial state preparation
# ============================================================

alpha_0   = 0.7
alpha_1   = 0.6
alpha_2   = 0.5
alpha_3   = 0.4

beta_0    = 0.4
beta_1    = 0.3
beta_2    = 0.2
beta_3    = 0.1


# ============================================================
# State preparation
# ============================================================

def prepare_initial_state(qc, wires):

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

    else:
        raise ValueError(
            f"Unsupported number of qubits = {num_qubits}"
        )