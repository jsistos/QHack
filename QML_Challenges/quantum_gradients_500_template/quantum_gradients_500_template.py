#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #

    ##GRADIENT CALCULATION
    ##
    gradient = np.zeros([6], dtype=np.float64)

    def parameter_shift_terms(qnode, params, pos):
        shifted = params.copy()

        shifted[pos] += np.pi / 2
        forward = qnode(shifted)

        shifted[pos] -= np.pi
        backward = qnode(shifted)

        return 0.5 * (forward - backward)

    for i in range(6):
        gradient[i] = parameter_shift_terms(qnode, params, i)

    ##
    ##GRADIENT CALCULATION
    
    ##BUILDING DAGGER OF VARIATIONAL CIRCUIT
    ##

    def non_parametrized_layer_dagger():
        qml.Hadamard(wires=0)
        qml.RZ(-b, wires=1)
        qml.CNOT(wires=[0, 1])
        qml.Hadamard(wires=1)
        qml.RZ(-a, wires=0)
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[0, 1])
        qml.RX(-a, wires=1)
        qml.RX(-b, wires=1)
        qml.RX(-a, wires=0)

    def variational_circuit_dagger(params):
        qml.RZ(-params[5], wires=2)
        qml.RY(-params[4], wires=1)
        qml.RX(-params[3], wires=0)
        non_parametrized_layer_dagger()
        qml.RZ(-params[2], wires=2)
        qml.RY(-params[1], wires=1)
        qml.RX(-params[0], wires=0)
        non_parametrized_layer_dagger()

    @qml.qnode(dev)
    def combined_circuits(shiftedParams, origParams):

        #Creates Psi_{\theta + \frac{\pi}{2}(e_i + e_j)
        variational_circuit(shiftedParams)

        #Creates Psi_{\theta}
        variational_circuit_dagger(origParams)

        return qml.probs(wires=[0,1,2])
    
    ##
    ##BUILDING DAGGER OF VARIATIONAL CIRCUIT

    ##CALCULATING FUBINI MATRIX
    ##
    def fubini_metric(posI, posJ):
        shifted = params.copy()
        
        shifted[posI] += np.pi / 2
        shifted[posJ] += np.pi / 2
        first = combined_circuits(shifted, params)[0]

        shifted[posJ] -= np.pi
        second = combined_circuits(shifted, params)[0]

        shifted[posI] -= np.pi
        fourth = combined_circuits(shifted, params)[0]

        shifted[posJ] += np.pi
        third = combined_circuits(shifted, params)[0]

        return 0.125 * (-first + second + third - fourth)

    fubini_matrix = np.zeros((6,6))
    
    for i in range(6):
        for j in range(6):
            fubini_matrix[i][j] = fubini_metric(i, j)

    fubini_inverse = np.linalg.inv(np.matrix(fubini_matrix))

    ##
    ##CALCULATING FUBINI MATRIX

    natural_grad = np.dot(fubini_inverse, gradient)

    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
