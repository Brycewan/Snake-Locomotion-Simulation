import numpy as np
import utils
import parameters

def val(x, e, l2, k, seg_index, time, wave_speed, amplitude, wave_length, v):
    sum = 0.0
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        # muscle contraction
        contraction = 1.0
        # left_contraction = 1.0
        # right_contraction = 1.0
        if seg_index[i][0] == True:
            offset = muscle_contraction(seg_index[i][1], time, wave_speed, amplitude, wave_length)
            # rectilinear
            # contraction += offset
            # if contraction < 0:
            #     contraction = 0
            # diff *= contraction
            
            # horizontal undulatory 
            # left contraction
            if seg_index[i][2] == 0 or seg_index[i][2] == 2:
                contraction += offset
                if contraction < 0:
                    contraction = 0
                diff *= contraction
            # right contraction
            elif seg_index[i][2] == 1 or seg_index[i][2] == 3:
                contraction -= offset
                if contraction < 0:
                    contraction = 0
                diff *= contraction
            
        spring_energy = l2[i] * 0.5 * k[i] * (diff.dot(diff) / l2[i] - 1) ** 2
        
        # damping  
        # h = parameters.TIME_STEP
        # damping_diff = ((x[e[i][0]] + v[e[i][0]] * h) - (x[e[i][1]] + v[e[i][1]] * h)) - diff
        # damping_energy = 0.5 * parameters.DAMPING_COEFF / (h**2) * damping_diff.dot(damping_diff)
    
        sum += spring_energy
    return sum

def grad(x, e, l2, k, seg_index, time, wave_speed, amplitude, wave_length, v):
    g = np.array([[0.0, 0.0, 0.0]] * len(x))
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        
        # muscle contraction
        contraction = 1.0
        left_contraction = 1.0
        right_contraction = 1.0
        if seg_index[i][0] == True:
            offset = muscle_contraction(seg_index[i][1], time, wave_speed, amplitude, wave_length)
            # rectilinear
            # contraction += offset
            # if contraction < 0:
            #     contraction = 0
            # diff *= contraction
            
            # horizontal undulatory 
            # left contraction
            if seg_index[i][2] == 0 or seg_index[i][2] == 2:
                contraction += offset
                if contraction < 0:
                    contraction = 0
                diff *= contraction
            # right contraction
            elif seg_index[i][2] == 1 or seg_index[i][2] == 3:
                contraction -= offset
                if contraction < 0:
                    contraction = 0
                diff *= contraction
            
        g_diff = 2 * k[i] * (diff.dot(diff) / l2[i] - 1) * diff * contraction
        
        # damping
        # vel_diff = v[e[i][0]] - v[e[i][1]]
        # damping_force = -parameters.DAMPING_COEFF * vel_diff
        
        g[e[i][0]] += g_diff
        g[e[i][1]] -= g_diff
    return g

def hess(x, e, l2, k, seg_index, time, wave_speed, amplitude, wave_length, v):
    IJV = [[0] * (len(e) * 36), [0] * (len(e) * 36), np.array([0.0] * (len(e) * 36))]
    for i in range(0, len(e)):
        diff = x[e[i][0]] - x[e[i][1]]
        
        # muscle contraction
        contraction = 1.0
        left_contraction = 1.0
        right_contraction = 1.0
        if seg_index[i][0] == True:
            offset = muscle_contraction(seg_index[i][1], time, wave_speed, amplitude, wave_length)
            # rectilinear            
            # contraction += offset
            # if contraction < 0:
            #     contraction = 0
            # diff *= contraction
            
            # horizontal undulatory 
            # left contraction
            if seg_index[i][2] == 0 or seg_index[i][2] == 2:
                contraction += offset
                if contraction < 0:
                    contraction = 0
                diff *= contraction
            # right contraction
            elif seg_index[i][2] == 1 or seg_index[i][2] == 3:
                contraction -= offset
                if contraction < 0:
                    contraction = 0
                diff *= contraction
        
        H_diff = 2 * k[i] * (contraction ** 2) / l2[i] * (2 * np.outer(diff, diff) + (diff.dot(diff) - l2[i]) * np.identity(3))
        
        # damping
        # H_damping = parameters.DAMPING_COEFF * np.identity(3)
        # H_local = utils.make_PSD(np.block([[H_spring + H_damping, -(H_spring + H_damping)], 
        #                             [-(H_spring + H_damping), H_spring + H_damping]]))
        
        H_local = utils.make_PSD(np.block([[H_diff, -H_diff], [-H_diff, H_diff]]))
        # add to global matrix
        for nI in range(0, 2):
            for nJ in range(0, 2):
                indStart = i * 36 + (nI * 2 + nJ) * 9
                for r in range(0, 3):
                    for c in range(0, 3):
                        IJV[0][indStart + r * 3 + c] = e[i][nI] * 3 + r
                        IJV[1][indStart + r * 3 + c] = e[i][nJ] * 3 + c
                        IJV[2][indStart + r * 3 + c] = H_local[nI * 3 + r, nJ * 3 + c]
    return IJV

def muscle_contraction(seg_index, time, wave_speed, amplitude, wave_length):
    dir = 0.0 * np.pi
    theta = 2 * np.pi * (seg_index / wave_length - (wave_speed/wave_length) * time)
    offset = amplitude * np.cos(theta - dir)
    return offset

