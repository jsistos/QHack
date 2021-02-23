#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

num_wires = 3
dev = qml.device("default.qubit", wires = num_wires)

a = 0.7
b = -0.3

params = [1,1,1,2,2,2]

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

def non_parametrized_layer():
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

def variational_dagger(params):
    qml.RZ(-params[5], wires=2)
    qml.RY(-params[4], wires=1)
    qml.RX(-params[3], wires=0)
    non_parametrized_layer_dagger()
    qml.RZ(-params[2], wires=2)
    qml.RY(-params[1], wires=1)
    qml.RX(-params[0], wires=0)
    non_parametrized_layer_dagger()

def variational_circuit(params):
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

@qml.qnode(dev)
def tryfunc():
    variational_dagger(params)
    variational_circuit(params)
    
    return qml.probs(wires=[0,1, 2])

print(tryfunc())