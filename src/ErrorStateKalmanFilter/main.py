from ErrorStateKalmanFilter.data import Data
from ErrorStateKalmanFilter.filter import ESKF
data = Data.get_data(IMU_data.csv)

def run_filter(data):
    for i in len(data)
        
    filter = ESKF.predict(data)



































"""
LIVE FILTER

This module will contain the live filtering loop that
will call the propagation and filter update steps.

import os
import numpy as np
import pandas as pd
from filter import predict, update
from propagation import propagate_nominal_state, imu_inputs, get_u_corrected, get_sigmas
imu_data = pd.read_csv('IMU_data.csv')


INITIALIZATION OF STATE VARIABLES FOR PROPOGATION
p = position
v = velocity
q = orientation quaternion
ab = accelerometer bias
wb = gyroscope bias
gt = gravity vector

dt = imu_data['Time(ms)'].iloc[1] - imu_data['Time(ms)'].iloc[0]
ax = imu_data['AccX(m/s^2)'].iloc[0]
ay = imu_data['AccY(m/s^2)'].iloc[0]
az = imu_data['AccZ(m/s^2)'].iloc[0]
wx = imu_data['GyroX(rad/s)'].iloc[0]
wy = imu_data['GyroY(rad/s)'].iloc[0]
wz = imu_data['GyroZ(rad/s)'].iloc[0]
mag_x = imu_data['MagX(uT)'].iloc[0]
mag_y = imu_data['MagY(uT)'].iloc[0]
mag_z = imu_data['MagZ(uT)'].iloc[0]
roll, pitch, yaw = imu_data['Roll(deg)'].iloc[0], imu_data['Pitch(deg)'].iloc[0], imu_data['Yaw(deg)'].iloc[0]
px, py, pz = 0.0, 0.0, 0.0
vx, vy, vz = 0.0, 0.0, 0.0
qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
axb, ayb, azb = 0.0, 0.0, 0.0
wxb, wyb, wzb = 0.0, 0.0, 0.0
gx, gy, gz = 0.0, 0.0, 9.81

p = np.array([[px], [py], [pz]])
v = np.array([[vx], [vy], [vz]])
q = np.array([[qw], [qx], [qy], [qz]])
ab = np.array([[ax], [ay], [az]])
wb = np.array([[wx], [wy], [wz]])
g = np.array([[gx], [gy], [gz]])


"""
IMU INPUT VARIABLES
dt = time step
ax, ay, az = accelerometer measurements
wx, wy, wz = gyroscope measurements
mag_x, mag_y, mag_z = magnetometer measurements
roll, pitch, yaw = euler angles from IMU
"""
am = np.array([[ax], [ay], [az]])
wm = np.array([[wx], [wy], [wz]])
magm = np.array([[mag_x], [mag_y], [mag_z]])
euler_angles = np.array([[roll], [pitch], [yaw]])
sensor_data = np.vstack((am, wm, magm, euler_angles))

"""
NOISE VECOTR
sigma_a_noise: magnitude std of aw
sigma_w_noise: magnitude std of ww
sigma_a_walk: magnitude std of a_walk
sigma_w_walk: magnitude std of w_walk
"""
sigma_a_noise = 0.01
sigma_w_noise = 0.001
sigma_a_walk = 0.0001
sigma_w_walk = 0.0001


def main():
    total_time = 0.0
    i = 0
    logs = []  # Initialize log list

    x = propagate_nominal_state(p, v, q, ab, wb, g)
    P = np.eye(18)  # Initial error covariance matrix
    U = imu_inputs(am, wm, magm, euler_angles)
    w = get_sigmas(sigma_a_noise, sigma_w_noise, sigma_a_walk, sigma_w_walk)
    V = np.diag([
        0.0001, 0.0001, 0.0001,        # accelerometer x,y,z
        0.000001, 0.000001, 0.000001,  # gyroscope x,y,z
        0.01                             # altimeter
    ])

    while True:
        dt = imu_data['Time(ms)'].iloc[i+1] - imu_data['Time(ms)'].iloc[i]
        ax = imu_data['AccX(m/s^2)'].iloc[i]
        ay = imu_data['AccY(m/s^2)'].iloc[i]
        az = imu_data['AccZ(m/s^2)'].iloc[i]
        wx = imu_data['GyroX(rad/s)'].iloc[i]
        wy = imu_data['GyroY(rad/s)'].iloc[i]
        wz = imu_data['GyroZ(rad/s)'].iloc[i]
        mag_x = imu_data['MagX(uT)'].iloc[i]
        mag_y = imu_data['MagY(uT)'].iloc[i]
        mag_z = imu_data['MagZ(uT)'].iloc[i]
        roll, pitch, yaw = imu_data['Roll(deg)'].iloc[i], imu_data['Pitch(deg)'].iloc[i], imu_data['Yaw(deg)'].iloc[i]

        # Predict step
        x, P = predict(x, P, U, w, dt)
        total_time += dt
        print(f'Predicted orientation at time t={total_time} :', x[6:10].flatten())

        # Update step
        x, P = update(sensor_data, x, P, V, dt, altimeter_data=False, gps_data=False,
                      use_pos=False, use_accel=False, use_magno=False, use_alti=True)
        print(f'Updated orientation at time t={total_time} :', x[6:10].flatten())

        # Append to logs
        logs.append([total_time] + x[6:10].tolist())

        i += 1
        
        if i >= len(imu_data) - 1:
           break

    # Convert logs to DataFrame and save CSV
    #df_logs = pd.DataFrame(logs, columns=['time_ms', 'q0', 'q1', 'q2', 'q3'])
    #df_logs.to_csv('orientation_log.csv', index=False)

if __name__ == "__main__":
    main()

"""