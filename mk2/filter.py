import numpy as np
from pyquaternion import Quaternion
import time
from propagation import (
    propogate_nominal_state,
    imu_inputs,
    get_u_corrected
    get_process_noise
)

"""
FILTER.PY

This module will contain the prediction step and the
update step of the ESKF filter.
"""

def skew_symmetric(v):
    vx, vy, vz = v
    return np.array([[0, -vz, vy],
                     [vz, 0, -vx],
                     [-vy, vx, 0]])

def prediction_step(x, P, U, w):
    """
    PREDICTION STEP FOR THE FILTER
    x: [p, v, q, ab, wb, g] state vector
    U: [ax ay az wx wy wz dt]  IMU body input vector
    w: [sigma_a_noise, sigma_w_noise, sigma_a_walk, sigma_w_walk] process noise vector
    """
    #gets the quaternion Orientation from the state vector q
    orientation = Quaternion(array=x[6:10])
    #creates a matrix so that I can rotate the acceleration from body to global frame
    R = orientation.rotation_matrix

    #a_global/a_true equation
    a_global = R @ (U[0:3] - x[10:13]) - x[-1:]

    #plug in a_global into the kinematic equations to set position adn velocitt according to the new orientation
    p_next = x[:3] + x[3:6] + 0.5*a_global*(x[-1:]**2)
    v_next = x[3:6] + a_global*x[-1:]

    #Here the reason why we get the 3d theta to just one theta is because
    #we want everythin in one step. if we passed in all three if would go in order but that 
    #would be wrong because the first rotation would change the axis for the second rotation
    #thus we combine them into one rotation vector
    three_dimensional_theta = (U[3:6]-x[13:16])*x[-1:]
    theta = norm(three_dimensional_theta)

    #this normlizes the axis. Tilts in the actual direction its going
    if theta > 0:
        axis = three_dimensional_theta / theta
    else:
        axis = [1, 0, 0]  #default axis if no rotation

    delta_q = Quaternion(axis=axis , angle=theta).normalised
    q_next = (Quaternion(array=x[6:10])

    x[0:3] = p_next
    x[3:6] = v_next
    x[6:10] = q_next.elements

    I = np.eye(15)
    F_i = np.array([[0, 0, 0, 0],
                  [I, 0, 0, 0],
                  [0, I, 0, 0],
                  [0, 0, I, 0],
                  [0, 0, 0, I],
                  [0, 0, 0, 0]])

    V_i = sigma_a_noise * dt**2 * I  
    theta_i = sigma_w_noise * dt**2 * I
    A_i = sigma_a_walk**2 * dt * I
    omega_i = sigma_w_walk**2 * dt * I


    i = np.array([[V_1],
                  [theta_i],
                  [A_i],
                  [omega_i]])

    Q_i = np.array([[V_i, 0, 0, 0],
                   [0, theta_i, 0, 0],
                   [0, 0, A_i, 0],
                   [0, 0, 0, omega_i]])

    rsa = -R @ skew_symmetric(U[0:3] - x[10:13]) * dt
    rsg = R.T @ skew_symmetric(U[3:6] - x[13:16]) * dt

    F_x = np.array([[I, I*dt,    0,  0, 0,    0],
                   [0, I,  rsa, -R, 0,    I*dt],
                   [0, 0,  rsg,  0, -I*dt,   0],
                   [0, 0,    0,  I,     0,   0],
                   [0, 0, 0, 0,         I,   0],
                   [0, 0, 0, 0,     0,       I]])

    P = Fx @ P @ Fx.T + F_i @ Q_i @ F_i.T

    return x, P