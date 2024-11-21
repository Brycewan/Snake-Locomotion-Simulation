import copy
from cmath import inf

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

import InertiaEnergy
import MassSpringEnergy
import GravityEnergy
import parameters
import debug

def step_forward(x, e, v, m, l2, k, h, tol, time, seg_index, wave_speed, amplitude, wave_length):
    x_tilde = x + v * h     # implicit Euler predictive position
    x_n = copy.deepcopy(x)
    
    # Newton loop
    iter = 0
    E_last = IP_val(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v)
    p = search_dir(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v)
    while LA.norm(p, inf) / h > tol:
        print('Iteration', iter, ':')
        print('residual =', LA.norm(p, inf) / h)

        # line search
        alpha = 1
        while IP_val(x + alpha * p, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v) > E_last:
            alpha /= 2
        print('step size =', alpha)

        x += alpha * p
        E_last = IP_val(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v)
        p = search_dir(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v)
        iter += 1
    
    v = (x - x_n) / h   # implicit Euler velocity update
    # Directional Friction
    for p in range(4, len(x) - 4, 4):
        # Compute the local forward spine unit vector
        last_points = [x[p - 4], x[p - 3], x[p - 2], x[p - 1], x[p], x[p + 1], x[p + 2], x[p + 3]]
        next_points = [x[p], x[p + 1], x[p + 2], x[p + 3], x[p + 4], x[p + 5], x[p + 6], x[p + 7]]
        centre_last = np.mean(last_points, axis=0)
        centre_next = np.mean(next_points, axis=0)
        
        spine_vec = (centre_last - centre_next) / LA.norm(centre_last - centre_next)
        
        # Adjust the velocity based on the dot product condition
        left_top_dot_product = np.dot(spine_vec, v[p])
        right_top_product = np.dot(spine_vec, v[p + 1])
        left_bottom_dot_product = np.dot(spine_vec, v[p + 2])
        right_bottom_product = np.dot(spine_vec, v[p + 3])
        
        if left_top_dot_product < 0:
            v[p] -= spine_vec * left_top_dot_product * parameters.MU
        if right_top_product < 0:
            v[p + 1] -= spine_vec * right_top_product * parameters.MU
        if left_bottom_dot_product < 0:
            v[p + 2] -= spine_vec * left_bottom_dot_product * parameters.MU
        if right_bottom_product < 0:
            v[p + 3] -= spine_vec * right_bottom_product * parameters.MU

    return [x, v]

def IP_val(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v):
    # return InertiaEnergy.val(x, x_tilde, m) + h * h * (MassSpringEnergy.val(x, e, l2, k) + GravityEnergy.val(x, m))     # implicit Euler
    return InertiaEnergy.val(x, x_tilde, m) + h * h * MassSpringEnergy.val(x, e, l2, k, seg_index, time, wave_speed, amplitude, wave_length, v)

def IP_grad(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v):
    # return InertiaEnergy.grad(x, x_tilde, m) + h * h * (MassSpringEnergy.grad(x, e, l2, k) + GravityEnergy.grad(x, m))    # implicit Euler
    return InertiaEnergy.grad(x, x_tilde, m) + h * h * MassSpringEnergy.grad(x, e, l2, k, seg_index, time, wave_speed, amplitude, wave_length, v)

def IP_hess(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v):
    IJV_In = InertiaEnergy.hess(x, x_tilde, m)
    IJV_MS = MassSpringEnergy.hess(x, e, l2, k, seg_index, time, wave_speed, amplitude, wave_length, v)
    IJV_MS[2] *= h * h    # implicit Euler
    IJV = np.append(IJV_In, IJV_MS, axis=1)
    H = sparse.coo_matrix((IJV[2], (IJV[0], IJV[1])), shape=(len(x) * 3, len(x) * 3)).tocsr()
    return H

def search_dir(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v):
    projected_hess = IP_hess(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v)
    # debug.compute_eigenvalues(projected_hess)
    reshaped_grad = IP_grad(x, e, x_tilde, m, l2, k, h, seg_index, time, wave_speed, amplitude, wave_length, v).reshape(len(x) * 3, 1)
    return spsolve(projected_hess, -reshaped_grad).reshape(len(x), 3)

