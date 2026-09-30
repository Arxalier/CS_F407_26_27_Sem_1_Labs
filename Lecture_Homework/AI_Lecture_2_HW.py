"""
Work through this by hand first, then in code.
1. Code the worked example from scratch in NumPy and
reproduce y^, L and all nine gradients.
2. Rebuild the same network in PyTorch and check .grad against
your NumPy values.
3. Run a finite-difference check on each of the nine parameters.
4. Train for a few hundred steps and plot L against step number.
5. Keep the sigmoid output unit, but replace the hidden
activation sigma first with tanh and then with ReLU. Report
∥∂L/∂a∥2 in each case. Which hidden activation gives the
largest gradient norm for these particular values, and why?
You won't need a GPU for this. The whole notebook will run
locally on a CPU within a minute
"""
# Placeholder values will be used for this homework
# L = 1/2 (y^ - y)^2
# delD/dely^ = (y^-y)
# y^ = sigma(z),
# delL/delZ = (y^-y)*y^(1-y^)
# dW2 = delL/delW2 = dx.h^T
# db2 = delL/delB2 = delL/delZ
# dh = delL/delh = W2.T delL/delZ
# da = delL/dela = dh (hadamard) h(1-h)
# dW1 = delL/delW1 = da @ x.T
# db1 = delL/delb1 = da

import numpy as np
import matplotlib.pyplot as plt

h = np.array([[0.6225], [0.3430]])
W2 = np.array([
    [0.3, -0.5],
    [-0.2, 0.4],
    [0.1, 0.2]
])
b2 = np.array([
    [0.2],
    [0.0],
    [-0.1]
])
y = np.array([[1.0],
              [0.0],
              [0.0]])

def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return exp_z/np.sum(exp_z, axis=0)

def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_derivative(x): 
    s = sigmoid(x)
    return s*(1-s)
def tanh(x): return np.tanh(x)
def tanh_derivative(x): return 1.0-tanh(x)**2
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return np.where(x>0, 1.0, 0.0)
# How does np.where work? it is like a ternary
# condition? param1: param2
z = W2@h + b2
p = softmax(z)
L = -np.sum(y*np.log(p+1e-9)) # to prevent log of 0
print("Forward pass values")
print(f"z (logits):\n{z.round(4)}\n")
print(f"probabilities:\n{p.round(4)}\n")
print(f"Loss L: {L.round(4)}\n")

dz = p-y
dW2 = dz @ h.T
db2 = dz

dW2 = dz @ h.T
db2 = dz

print("Backward pass values")
print(f"dL/dz (p-y):\n{dz.round(5)}\n")
print(f"dW2:\n{dW2.round(4)}")