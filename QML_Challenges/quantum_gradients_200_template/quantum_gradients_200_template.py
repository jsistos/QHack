#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    # Execute unshifted circuit for later use
    unshifted = circuit(weights)

    #Obtain partial derivative didj with an angle of pi/2
    def parameter_shift_double(qnode, params, posI, posJ):
        shifted = params.copy()

        shifted[posI] += np.pi / 2
        shifted[posJ] += np.pi / 2
        fForward = qnode(shifted)

        shifted[posI] -= np.pi
        iBackward = qnode(shifted)

        shifted[posJ] -= np.pi
        fBackward = qnode(shifted)

        shifted[posI] += np.pi
        jBackward = qnode(shifted)

        #Formula is obtained from applying partial derivative formula twice and s = pi/2
        return 0.25 * ((fForward + fBackward) - (iBackward + jBackward))

    #Obtain partial derivative dx2 with an angle of pi/4. Useful to reuse hessian results
    # in gradient vector later
    def parameter_shift_squared(qnode, params, pos):
        shifted = params.copy()

        shifted[pos] += np.pi / 2
        forward = qnode(shifted)

        shifted[pos] -= np.pi
        backward = qnode(shifted)

        #Formula is obtained from partial derivative twice and s = pi/4. Note the unshifted used here
        return [0.5 * (forward + backward - 2*unshifted), 0.5 * (forward - backward)]

    # Hessian diagonal partial d2f / dx2
    for i in range(5):
        #Shift parameter method. Save to variable to avoid executing twice
        partialDiffSquared = parameter_shift_squared(circuit, weights, i)

        #Extract hessian value from execution
        hessian[i][i] = partialDiffSquared[0]
        
        #Extract gradient value from execution
        gradient[i] = partialDiffSquared[1]
    
    # Hessian fill non diagonal partial d2f / dxdy
    for i in range(5):
        for j in range(i):
            #Shift parameter method. Save to variable to avoid executing twice
            partialDiffDouble = parameter_shift_double(circuit, weights, i, j)

            #Assign result to (i, j) and (j, i). They're equal because of the symmetry
            hessian[i][j] = partialDiffDouble
            hessian[j][i] = partialDiffDouble


    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
