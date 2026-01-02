import math
from typing import Tuple
import numpy as np
from numpy.linalg import norm
from pyquaternion import Quaternion
import pandas as pd
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_SENSOR_PATH = os.path.join(HERE, 'sensor_data.csv')
CSV_MOTION_PATH = os.path.join(HERE, 'motion_data.csv')
sensor_data = pd.read_csv(CSV_SENSOR_PATH)
motion_data = pd.read_csv(CSV_MOTION_PATH)

class Data:
    def __init__(self):
        self.TimeCol = 0
        self.AccXCol = 1
        self.AccYCol = 2
        self.AccZCol = 3
        self.GyroXCol = 4
        self.GyroYCol = 5
        self.GyroZCol = 6
        self.PosX = 0
        self.PosY = 1
        self.PosZ = 2
        self.VelX = 3
        self.VelY = 4
        self.VelZ = 5
        self.sensor_data = sensor_data
        self.motion_data = motion_data

    def get_imu_vectors(self, i):
        gyro = np.array([
            self.sensor_data.iloc[i, self.GyroXCol],
            self.sensor_data.iloc[i, self.GyroYCol],
            self.sensor_data.iloc[i, self.GyroZCol]
        ])
        acc = np.array([
            self.sensor_data.iloc[i, self.AccXCol],
            self.sensor_data.iloc[i, self.AccYCol],
            self.sensor_data.iloc[i, self.AccZCol],
        ])
        dt = self.motion_data.iloc[i + 1, self.TimeCol] - self.motion_data.iloc[i, self.TimeCol]
        return gyro, acc, dt
    
    def get_motion_vectors(self, i):
        Pos = np.array([
            self.motion_data.iloc[i, self.PosX],
            self.motion_data.iloc[i, self.PosY],
            self.motion_data.iloc[i, self.PosZ]
        ])
        Vel = np.array([
            self.motion_data.iloc[i, self.VelX],
            self.motion_data.iloc[i, self.VelY],
            self.motion_data.iloc[i, self.VelZ]
        ])
        return Pos, Vel

class ESKF:
    def __init__(self, 
                data_obj,
                sig_a_noise=0.1,   #ACCELERATION NOISE PARAMTER
                sig_a_walk=0.1,    #ACCELERATION WALK PARAMETER
                sig_w_noise=0.1,   #GYRO NOISE PARAMETER
                sig_w_walk=0.1,    #GYRO WALK PARAMTER
                gravity=9.81):
        self.iteration = 0
        self.data_obj = data_obj
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
        self.gyro, self.acc, self.dt = self.data_obj.get_imu_vectors(self.iteration)  # Use property
        self.pos, self.vel = self.data_obj.get_motion_vectors(self.iteration)
        self.measurement = np.hstack((self.pos, self.vel))
        self.U = np.hstack((self.acc, self.gyro))
        self.R = None
        #self.results = [] ~~ for csv loading only

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

    def update(self, R_measurement=None):
        self.measurement = np.asarray(self.measurement).flatten()
        meas_size = len(self.measurement)

        p = self.X[0:3]
        v = self.X[7:10]
        
        # Default measurement noise if not provided
        if R_measurement is None:
            R_measurement = np.eye(meas_size) * 0.1

        if meas_size == 3:  # FIXED: added this case (was missing)
            # Position only
            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)
            y = self.measurement.reshape(3, 1) - p.reshape(3, 1)
        elif meas_size == 6:  # FIXED: was only handling 6D case
            # Position and velocity
            H = np.zeros((6, 15))
            H[0:3, 0:3] = np.eye(3)
            H[3:6, 6:9] = np.eye(3)
            predicted = np.vstack([p.reshape(3, 1), v.reshape(3, 1)])
            y = self.measurement.reshape(6, 1) - predicted
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

        self.results.append({
            'time': time.time(),
            'qw': self.X[3],
            'qx': self.X[4],
            'qy': self.X[5],
            'qz': self.X[6]
        })
        

        print(f"Time: {time.time():.6f} | Quaternion: [{self.X[3]:.6f}, {self.X[4]:.6f}, {self.X[5]:.6f}, {self.X[6]:.6f}]")
    
    #def save_results(self, filename='quaternion_results.csv'):
    #    """Save results to CSV"""
    #    import pandas as pd
    #    df = pd.DataFrame(self.results)
    #    df.to_csv(filename, index=False)
    #    print(f"Results saved to {filename}")
