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

def quat_rot()

def prediction_step(x, U, w):
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
    q_next = (Quaternion(array=x[6:10]) * delta_q).normalised