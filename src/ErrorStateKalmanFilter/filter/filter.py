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
                 Data,
                 sig_a_noise=0.1, 
                 sig_a_walk=0.1, 
                 sig_w_nosie=0.1, 
                 sig_w_walk=0.1, 
                 gravity=9.81):
        self.X = np.zeros(16)
        self.X[3] = 1
        self.error_state = np.zeros(15)
        self.P = np.eye(15)
        self.Qi = np.diag([
            sig_a_noise**2, sig_a_noise**2, sig_a_noise**2, 
            sig_w_nosie**2, sig_w_nosie**2, sig_w_nosie**2, 
            sig_a_walk**2, sig_a_walk**2, sig_a_walk**2, 
            sig_w_walk**2, sig_w_walk**2, sig_w_walk**2, 
        ])
        self.gravity = np.array([0, 0, gravity])
        self.gyro, self.acc, self.dt = Data.get_imu_data
        self.U = np.hstack((self.acc, self.gyro))
        self.R = None

    def skew_symmetric(v):
        vx, vy, vz = v.flatten()
        return np.array([[0, -vz, vy],
                        [vz, 0, -vx],
                        [-vy, vx, 0]])

    def q_skew_symmetric(q):
        qw, qx, qy, qz = q.flatten()
        return np.array([
            [-qx, -qy, -qz],
            [qw, -qz, qy],
            [qz, qw, -qx],
            [-qy, qx, qw]
        ])

    def q_rot(three_dimensional_theta):
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
        Fx[3:6, 3:6] = np.eye(3) - self._skew_symmetric(w) * dt
        Fx[3:6, 12:15] = -np.eye(3) * dt
        Fx[6:9, 3:6] = -R @ self._skew_symmetric(a) * dt
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

        a_global = R @ (self.IMU_data[0:3] - self.X[10:13]) - self.gravity
        p_next = X[0:3] + X[7:10]*dt + 0.5*a_global*(dt**2)
        v_next  = X[7:10]dt + a_global*dt

        am = self.U[0:3]
        wm = self.U[3:6]
        ab = self.X[10:13]
        wb = self.X[13:16]

        a_unbiased = am - ab
        w_unbiased = wm - wb

        three_dimensional_theta = (U[3:6] - X[13:16])*dt
        delta_q = q_rot(three_dimensional_theta)
        q_next = (Quaternion(array=X[3:7])*delta_q).normalised

        X[0:3] = p_next
        X[3:7] = q_next.elements.reshape(4,1)
        X[7:10] = v_next

        Fx = compute_error_state_jacobian(self.dt, a_unbiased, w_unbiased, R)
        Fi = compute_noise_jacobian(self.dt, R)

        self.P = Fx @ self.P @ Fx.T @ + Fi @ self.Qi @ Fi.T
        self.error_state = np.zeros(15)

    def update(self, X, R=None):

        p = X[0:3]
        v = X[7:10]
        
        H = np.zeros((6, 15))
        H[0:3, 0:3] = np.eye(3)  # position
        H[3:6, 6:9] = np.eye(3)  # velocity
        predicted = np.vstack([p.reshape(3, 1), v.reshape(3, 1)])
        y = measurement.reshape(6, 1) - predicted

        # Kalman Gain
        S = H @ self.P @ H.T + R_measurement
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update error state
        self.error_state = (K @ y).flatten()


def update(imu_data, x, P, V, dt, altimeter_data, gps_data,
                use_pos=False, use_accel=False, use_magno=False, use_alti=True):
    """
    UPDATE STEP FOR THE FILTER
    imu_data: [am, wm, magm, euler_angles]
    x: [p, v, q, ab, wb, g] state vector
    P: error covariance matrix
    V: measurement noise covariance matrix
    dt: time step
    """
    #these are the measurements from the IMU, gps, and altimeter
    # Column vector version
    y_a = imu_data[0:3].reshape(-1,1)        # (3,1)
    y_m = imu_data[6:9].reshape(-1,1)        # (3,1)
    y_z = np.array([altimeter_data]).reshape(1,1) if altimeter_data is not None else None
    y_p = np.array(gps_data).reshape(-1,1) if gps_data is not None else None

    b = np.array([1, 0, 0]).reshape(-1,1)    # magnetic field column
    g = x[16:19].reshape(-1,1)               # gravity column

    orientation = Quaternion(array=x[6:10])
    R = orientation.rotation_matrix

    # Prediction vectors
    y_a_pred = (R.T @ g).reshape(-1,1)
    y_m_pred = (R.T @ b).reshape(-1,1)
    y_z_pred = np.array([x[2]]).reshape(1,1)

    if use_pos and (y_p is not None):
        y_p_pred = x[0:2].reshape(-1,1)
        y_pred = np.vstack((y_a_pred, y_m_pred, y_z_pred, y_p_pred))  # (9,1)
        y = np.vstack((y_a, y_m, y_z, y_p))                            # (9,1)
        dim = 9
        pos_mode = True
    else:
        y_pred = np.vstack((y_a_pred, y_m_pred, y_z_pred))             # (7,1)
        y = np.vstack((y_a, y_m, y_z))                                  # (7,1)
        dim = 7
        pos_mode = False


    # Create effective V matrix by inflating noise for disabled measurements
    # This is more theoretically correct than setting y_pred = y
    V_effective = V.copy()
    INF = 1e10  # Very large noise = filter ignores this measurement
    if not use_accel:
        V_effective[0:3, 0:3] *= INF
    if not use_magno:
        V_effective[3:6, 3:6] *= INF
    if use_pos and y_p is not None:
        # When using position: [accel(0:3), mag(3:6), pos(6:8), alt(8)]
        if not use_alti:
            V_effective[8, 8] *= INF
    else:
        # When not using position: [accel(0:3), mag(3:6), alt(6)]
        if not use_alti:
            V_effective[6, 6] *= INF

    
    #gets the skew matrix of gravity and magnetic field
    g_cross = skew_symmetric(g)
    b_cross = skew_symmetric(b)
    q = x[6:10]

    #jacobian matrix for measurement model
    H_a = -R.T @ g_cross
    H_m = -R.T @ b_cross
    H_ya_ma_wrt_q = np.vstack((H_a, H_m))
    Q_x = 0.5 * q_skew_symmetric(q)  # 4x3

    # Build H_x explicitly (measurement rows x true-state cols)
    true_n = 19
    err_n = P.shape[0]  # derive error-state dimension from P to avoid mismatches
    H_x = np.zeros((dim, true_n))

    # accel(3) and mag(3) rows depend on quaternion small-angle (maps into true-state q cols 6:10 -> use 6:9 for error)
    H_x[0:6, 6:9] = H_ya_ma_wrt_q  # (6 x 3)

    # measurement stacking in y_pred: [accel(3), mag(3), alt(1), pos(2)] when pos_mode True
    if pos_mode:
        # altimeter row is at index 6
        H_x[6, 2] = 1.0
        # position rows are at indices 7 and 8 (x,y)
        H_x[7:9, 0:2] = np.eye(2)
    else:
        # no position: altimeter row is at index 6
        H_x[6, 2] = 1.0

    # Build X_x (true-state -> error-state) explicitly
    X_x = np.zeros((true_n, err_n))
    # position & velocity map to first 6 error cols
    if err_n >= 6:
        X_x[0:6, 0:6] = np.eye(6)
    else:
        X_x[0:6, 0:err_n] = np.eye(6)[:, :err_n]

    # quaternion true->error mapping: rows 6:10 in true state, cols 6:9 in error state
    if err_n > 6:
        qcols = min(3, err_n - 6)
        X_x[6:10, 6:6+qcols] = Q_x[:, :qcols]

    # accel bias true indices 10:13 -> error cols 9:12
    if err_n > 9:
        bcols = min(3, err_n - 9)
        X_x[10:13, 9:9+bcols] = np.eye(3)[:, :bcols]

    # gyro bias true indices 13:16 -> error cols 12:15
    if err_n > 12:
        gcols = min(3, err_n - 12)
        X_x[13:16, 12:12+gcols] = np.eye(3)[:, :gcols]

    # gravity (true indices 16:19) not part of 15-element error state -> leave zeros

    # state-to-sensor measurement jacobian (dim x err_n)
    H = H_x @ X_x
    #Kalman Gain -> high = more trust in measurements, low = more trust in prediction
    # Kalman gain (use solve for stability)
    S = H @ P @ H.T + V_effective
    K = P @ H.T @ np.linalg.inv(S)
    #error gain computation 
    #inovation = measurement residual
    innovation = (y - y_pred)
    delta = K @ innovation
    # apply corrections to nominal state
    x[0:3] += delta[0:3]      # position
    x[3:6] += delta[3:6]      # velocity
    # update Quaternion with small angle approx.
    # q = q_old ⊗ δq
    dq = delta[6:9]
    x[6:10] = (Quaternion(array=x[6:10]) * q_rot(dq)).normalised.elements.reshape(4,1)
    x[10:13] += delta[9:12]   # acc bias
    x[13:16] += delta[12:15]  # gyro bias
    #Using joseph form to update P to ensure it remains symmetric positive definite
    P[:] = (np.eye(18) - K @ H)@P@(np.eye(18) - K@H).T + K@V_effective@K.T
    #making a new jacobian G to account for quaternion update
    G = np.eye(18)
    # this resets the error theta part of the quaternion after update
    G[6:9, 6:9] -= skew_symmetric((1/2)*dq)
    P[:]= G @ P @ G.T
    return x, P

