#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

WIRES = 2

hamiltonian = np.array([0.863327072347624,0.0167108057202516,0.07991447085492759,0.0854049026262154,0.0167108057202516,0.8237963773906136,-0.07695947154193797,0.03131548733285282,0.07991447085492759,-0.07695947154193795,0.8355417021014687,-0.11345916130631205,0.08540490262621539,0.03131548733285283,-0.11345916130631205,0.758156886827099])
hamiltonian = np.array(hamiltonian, float).reshape((2 ** WIRES, 2 ** WIRES))

# Generate random initial parameters
np.random.seed(1967)
initial_params = np.random.random(30)

print(hamiltonian)
print(initial_params)