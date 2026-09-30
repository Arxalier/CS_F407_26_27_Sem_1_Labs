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

x_val = np.array([[1.0], [-1.0]])
y_val = np.array([[1.0]])

W1_init = np.array([[0.5, -0.5],[0.3, 0.8]])
b1_init = np.array([[0.1], [-0.1]])
W2_init = np.array([[0.4, -0.6]])
b2_init = np.array([[0.2]])

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

print("Beginning the training loop")
W1,b1 = W1_init.copy(), b1_init.copy()
W2,b2 = W2_init.copy(), b2_init.copy()

learning_rate = 0.5
losses = []

for step in range(300):
    a = W1 @ x_val + b1
    h = sigmoid(a)
    z = W2 @ h + b2
    y_hat = sigmoid(z)

    loss = 0.5 * (y_hat-y_val)**2
    losses.append(loss.item()) # what does item method do?

    dy_hat = y_hat - y_val
    dz = dy_hat * sigmoid_derivative(z)

    dW2 = dz @ h.T
    db2 = dz
    dh = W2.T @ dz

    da = dh * sigmoid_derivative(a)
    dW1 = da @ x_val.T
    db1 = da

    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2

print(f"Final loss: {losses[-1]:.6f}")

plt.plot(losses)
plt.title("Loss across iterations")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.show()

def get_norm(hidden_activation, hidden_derivative):
    W1,b1 = W1_init.copy(), b1_init.copy()
    W2,b2 = W2_init.copy(), b2_init.copy()
    a = W1 @ x_val + b1
    h = hidden_activation(a)
    z = W2 @ h + b2
    y_hat = sigmoid(z)
    dy_hat = y_hat-y_val
    dz = dy_hat * sigmoid_derivative(z)
    dh = W2.T @ dz
    da = dh*hidden_derivative(a)
    return np.linalg.norm(da) # L2 norm

norm_sigmoid = get_norm(sigmoid, sigmoid_derivative)
norm_tanh = get_norm(tanh, tanh_derivative)
norm_relu = get_norm(relu, relu_derivative)

print(f"Norm of dL/da using sigmoid: {norm_sigmoid:.6f}")
print(f"Norm of dL/da using tanh: {norm_tanh:.6f}")
print(f"Norm of dL/da using relu: {norm_relu:.6f}")

