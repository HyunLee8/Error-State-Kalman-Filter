"""Implementation of an Error State Kalman Filter (ESKF) using quaternions for orientation representation."""
import math
import numpy as np
import pandas as pd
df = pd.read_csv('IMU_data.csv')  # Assuming a CSV file with sensor data

"""Nominal State Methods"""
def omega_matrix(omega):
    wx, wy, wz = omega.flatten()
    Omega = np.block([[0, -wx, -wy, -wz],
                      [wx, 0, wz, -wy],
                      [wy, -wz, 0, wx],
                      [wz, wy, -wx, 0]])
    return Omega

def quaternion_derivative(q, omega):
    Omega = omega_matrix(omega)
    q_dot = 0.5 * q @ Omega
    return q_dot

def update_bias():
    wxw = 1e-5
    wyw = 2e-5
    wzw = 1.5e-5
    process_noise = np.array([[wxw], [wyw], [wzw]])
    return process_noise

def bias_derivative(process_noise):
    return process_noise

def propogate_nominal_state(q, wb, wm, wn):
    process_noise = update_bias()
    q_dot = quaternion_derivative(q, wm - wb - wn)
    wb_dot = bias_derivative(process_noise)
    X_dot = np.vstack((q_dot, wb_dot))
    return X_dot

"""Error State Methods"""
#READER DONT WORRY ABOUT THIS (this is just to read my thoughts)
#u = wm - wb
#

def skew_symmetric(v):









if __name__ == "__main__":
    q = np.array([[1.0], [0.0],[0.0],[0.0]]) #default quaternion

    while count in range(len(df)-1):
        count = 1

        #INITIALIZAITION OF VARIABLES
        wx = df['GyroX(rad/s)'][count]
        wy = df['GyroY(rad/s)'][count]
        wz = df['GyroZ(rad/s)'][count]
        bx = np.zeros((3,1))
        by = np.zeros((3,1))
        bz = np.zeros((3,1))
        nx = np.zeros((3,1))
        ny = np.zeros((3,1))
        nz = np.zeros((3,1))


        wm = np.array([[wx], [wy], [wz]]) #measured angular velocity
        wb = np.array([[bx], [by], [bz]]) #bias for angular velocity
        wn = np.array([[nx], [ny], [nz]]) #noise for angular velocity

        X_dot = propogate_nominal_state(q, wb, wm, wn)