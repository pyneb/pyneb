from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as np


def func(x):
    return np.sin(x[0]) * np.sin(x[1])


x_value = np.array([0.0, 0.0])  # note inputs have to be floats
H_f = jacobian(egrad(func))  # returns a function
print(H_f(x_value))