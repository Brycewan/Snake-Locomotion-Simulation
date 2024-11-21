import numpy as np

def val(x, x_tilde, m):
    sum = 0.0
    for i in range(0, len(x)):
        diff = x[i] - x_tilde[i]
        sum += 0.5 * m[i] * diff.dot(diff)
    return sum

def grad(x, x_tilde, m):
    g = np.array([[0.0, 0.0, 0.0]] * len(x))
    for i in range(0, len(x)):
        g[i] = m[i] * (x[i] - x_tilde[i])
    return g

def hess(x, x_tilde, m):
    IJV = [[0] * (len(x) * 3), [0] * (len(x) * 3), np.array([0.0] * (len(x) * 3))]
    for i in range(0, len(x)):
        for d in range(0, 3):
            IJV[0][i * 3 + d] = i * 3 + d
            IJV[1][i * 3 + d] = i * 3 + d
            IJV[2][i * 3 + d] = m[i]
    return IJV