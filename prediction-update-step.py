"""WORK IN PROGRESS"""

import numpy as np
from propopating-nominal-error import propogate_nominal_state, propogate_error_state, Covariance, StateTransitionMatrix
from pyquaternion import Quaternion
import time

x = propagate_nominal_state(p, v, a, q, wb, wm, wn, ab, dt)
P = Covariance(P, dt)
U = [wm-wn, am-an]
dt = time_step

def skew_matrix(v):
    vx, vy, vz = v.flatten()
    return np.array([[0, -vz, vy],
                     [vz, 0, -vx],
                     [-vy, vx, 0]])

def 

def prediction_step(x, P, U, dt, Q):
    """
    x: Nominal State Vectors
    P: Error Covariance Matrix
    U: IMU body measurements
    dt: time step
    Q: Process Noise Covariance Matrix
    """



def update_step():