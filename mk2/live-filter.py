"""
LIVE FILTER

This module will contain the live filtering loop that
will call the propagation and filter update steps.
"""

import numpy as np
import pandas as pd
imu_data = pd.read_csv('IMU_data.csv')

"""
INITIALIZATION OF STATE VARIABLES FOR PROPOGATION
p = position
v = velocity
q = orientation quaternion
ab = accelerometer bias
wb = gyroscope bias
gt = gravity vector
"""
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
dt = imu_data['Time(ms)'].[i+1] - imu_data['Time(ms)'].[i]
ax = imu_data['AccX(m/s^2)'].[i]
ay = imu_data['AccY(m/s^2)'].[i]
az = imu_data['AccZ(m/s^2)'].[i]
wx = imu_data['GyroX(rad/s)'].[i]
wy = imu_data['GyroY(rad/s)'].[i]
wz = imu_data['GyroZ(rad/s)'].[i]
mag_x = imu_data['MagX(uT)'].[i]
mag_y = imu_data['MagY(uT)'].[i]
mag_z = imu_data['MagZ(uT)'].[i]
roll, pitch, yaw = imu_data['Roll(deg)'].[i], imu_data['Pitch(deg)'].[i], imu_data['Yaw(deg)'].[i]
am = np.array([[ax], [ay], [az]])
wm = np.array([[wx], [wy], [wz]])
magm = np.array([[mag_x], [mag_y], [mag_z]])
euler_angles = np.array([[roll], [pitch], [yaw]])
imu_data = np.vstack((am, wm, magm, euler_angles))

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


