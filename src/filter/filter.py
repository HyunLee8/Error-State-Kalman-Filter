import math
from typing import Tuple
import numpy as np
from numpy.linalg import norm
from pyquaternion import Quaternion
import time

"""
FILTER.PY
This module will contain the prediction step and the
update step of the ESKF filter.
"""

class ESKF:
    def __init__(self, 
                 sig_a_noise=0.1, 
                 sig_a_walk=0.1, 
                 sig_w_noise=0.1, 
                 sig_w_walk=0.1, 
                 gravity=9.81):
        self.iteration = 0
        self.X = np.zeros(16)
        self.X[3] = 1
        self.error_state = np.zeros(15)
        self.P = np.eye(15)
        self.Qi = np.diag([
            sig_a_noise**2, sig_a_noise**2, sig_a_noise**2, 
            sig_w_noise**2, sig_w_noise**2, sig_w_noise**2, 
            sig_a_walk**2, sig_a_walk**2, sig_a_walk**2, 
            sig_w_walk**2, sig_w_walk**2, sig_w_walk**2, 
        ])
        self.gravity = np.array([0, 0, gravity])
        self.gyro, self.acc, self.dt = Data.get_imu_data(itteration)
        self.U = np.hstack((self.acc, self.gyro))
        self.R = None

    def skew_symmetric(self, v):
        vx, vy, vz = v.flatten()
        return np.array([[0, -vz, vy],
                        [vz, 0, -vx],
                        [-vy, vx, 0]])

    def q_skew_symmetric(self, q):
        qw, qx, qy, qz = q.flatten()
        return np.array([
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw]
        ])

    def q_rot(self, three_dimensional_theta):
        theta = np.asarray(three_dimensional_theta).reshape(3)
        angle = np.linalg.norm(theta)
        #this normlizes the axis. Tilts in the actual direction its going
        if angle > 0:
            return Quaternion(axis=theta/angle, angle=angle)
        else:
            return Quaternion()
            #default axis if no rotation
            #creates error state quaternion. normalised because small angle approx

    def compute_noise_jacobian(self, dt, R):
        Fi = np.zeros((15, 12))
        Fi[6:9, 0:3] = -R * dt
        Fi[3:6, 3:6] = -np.eye(3) * dt
        Fi[9:12, 6:9] = np.eye(3) * dt
        Fi[12:15, 9:12] = np.eye(3) * dt
        return Fi

    def compute_error_state_jacobian(self, dt, a, w, R):
        Fx = np.eye(15)
        Fx[0:3, 6:9] = np.eye(3) * dt
        Fx[3:6, 3:6] = np.eye(3) - self.skew_symmetric(w) * dt
        Fx[3:6, 12:15] = -np.eye(3) * dt
        Fx[6:9, 3:6] = -R @ self.skew_symmetric(a) * dt
        Fx[6:9, 9:12] = -R * dt
        return Fx
        
    def predict(self):
        """
        Variables:
            X: [p, q, v ab, wb] state vector
            P: error covariance matrix
            U: [ax ay az wx wy wz]  IMU body input vector
            dt: time step
        """

        orientation = Quaternion(self.X[3:7])
        self.R = orientation.rotation_matrix

        am = self.U[0:3]
        wm = self.U[3:6]
        ab = self.X[10:13]
        wb = self.X[13:16]

        a_unbiased = am - ab
        w_unbiased = wm - wb

        a_global = self.R @ a_unbiased - self.gravity
        p_next = self.X[0:3] + self.X[7:10] * self.dt + 0.5 * a_global * (self.dt**2)
        v_next = self.X[7:10] + a_global * self.dt

        three_dimensional_theta = (w_unbiased)*self.dt
        delta_q = self.q_rot(three_dimensional_theta)
        q_next = (Quaternion(self.X[3:7])*delta_q).normalised

        self.X[0:3] = p_next
        self.X[3:7] = q_next.elements
        self.X[7:10] = v_next

        Fx = self.compute_error_state_jacobian(self.dt, a_unbiased, w_unbiased, self.R)
        Fi = self.compute_noise_jacobian(self.dt, self.R)

        self.P = Fx @ self.P @ Fx.T + Fi @ self.Qi @ Fi.T
        self.error_state = np.zeros(15)

        self.iteration = self.iteration + 1

    def update(self, measurement, R_measurement=None):
        measurement = np.asarray(measurement).flatten()
        meas_size = len(measurement)

        p = self.X[0:3]
        v = self.X[7:10]
        
        # Default measurement noise if not provided
        if R_measurement is None:
            R_measurement = np.eye(meas_size) * 0.1

        if meas_size == 3:  # FIXED: added this case (was missing)
            # Position only
            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)
            y = measurement.reshape(3, 1) - p.reshape(3, 1)
        elif meas_size == 6:  # FIXED: was only handling 6D case
            # Position and velocity
            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 6:9] = np.eye(3)
            predicted = np.vstack([p.reshape(3, 1), v.reshape(3, 1)])
            y = measurement.reshape(6, 1) - predicted
        else:
            raise ValueError(f"Measurement size {meas_size} not supported. Use 3 or 6.")

        # Kalman Gain
        S = H @ self.P @ H.T + R_measurement
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update error state
        self.error_state = (K @ y).flatten()

        # Update error covariance (Joseph form)
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_measurement @ K.T

        δp = self.error_state[0:3]
        δθ = self.error_state[3:6]
        δv = self.error_state[6:9]
        δab = self.error_state[9:12]
        δwb = self.error_state[12:15]

        self.X[0:3] += δp

        # Update quaternion
        q_nominal = Quaternion(self.X[3:7])
        angle = np.linalg.norm(δθ)
        
        if angle < 1e-8:
            δq = Quaternion(scalar=1.0, vector=0.5 * δθ)
        else:
            δq = Quaternion(axis=δθ/angle, angle=angle)
        
        q_updated = (q_nominal * δq).normalised
        self.X[3:7] = q_updated.elements
        
        # Update velocity
        self.X[7:10] += δv
        
        # Update biases
        self.X[10:13] += δab
        self.X[13:16] += δwb

        # Reset error state to zero
        self.error_state = np.zeros(15)

        print(f"Time: {time.time():.6f} | Quaternion: [{self.X[3]:.6f}, {self.X[4]:.6f}, {self.X[5]:.6f}, {self.X[6]:.6f}]")
