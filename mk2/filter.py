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

def q_skew_symmetric(q):
    qw, qx, qy, qz = q
    return np.array([[-qx, -qy, -qz],
                     [qw, -qz, qy],
                     [qz, qw, -qx],
                     [-qy, qx, qw]])

def q_rot(three_dimensional_theta):
    angle = norm(three_dimensional_theta)
    #this normlizes the axis. Tilts in the actual direction its going
    if theta > 0:
        return Quaternion(axis = theta/angle, angle=angle)
    else:
        axis = [1, 0, 0] 
        return Quaternion()
        #default axis if no rotation
        #creates error state quaternion. normalised because small angle approx
    

def prediction_step(x, P, U, w, dt):
    """
    PREDICTION STEP FOR THE FILTER
    x: [p, v, q, ab, wb, g] state vector
    P: error covariance matrix
    U: [ax ay az wx wy wz]  IMU body input vector
    w: [sigma_a_noise, sigma_w_noise, sigma_a_walk, sigma_w_walk] process noise vector
    dt: time step
    """
    #gets the quaternion Orientation from the state vector q
    orientation = Quaternion(array=x[6:10])
    #creates a matrix so that I can rotate the acceleration from body to global frame
    R = orientation.rotation_matrix

    #a_global/a_true equation
    a_global = R @ (U[0:3] - x[10:13]) - x[16:-1]

    #plug in a_global into the kinematic equations to set position adn velocitt according to the new orientation
    p_next = x[:3] + x[3:6] + 0.5*a_global*(dt**2)
    v_next = x[3:6] + a_global*dt
    #Here the reason why we get the 3d theta to just one theta is because
    #we want everythin in one step. if we passed in all three if would go in order but that 
    #would be wrong because the first rotation would change the axis for the second rotation
    #thus we combine them into one rotation vector
    three_dimensional_theta = (U[3:6]-x[13:16])*dt
    delta_q = q_rot(three_dimensional_theta)
    q_next = (Quaternion(array=x[6:10]) * delta_q).normalised
    #updating the state vector with the new position, velocity, and orientation; lets go!
    x[0:3] = p_next

    x[3:6] = v_next

    x[6:10] = q_next.elements

    #I is the identity matrix
    I = np.eye(3)

    #my dumaass put just normal 0's before i made this lol
    Z3 = np.zeros((3,3))

    #JACOBIAN MATRIX for noise
    F_i = np.array([[Z3, Z3, Z3, Z3],
                    [I, Z3, Z3,  Z3],
                    [Z3, I, Z3,  Z3],
                    [Z3, Z3, I,  Z3],
                    [Z3, Z3, Z3 , I],
                    [Z3, Z3, Z3, Z3]])

    #PROCESS NOISE COVARIANCE MATRIX 
    V_i = w[1:2] * dt**2 * I  
    theta_i = w[2:3] * dt**2 * I
    A_i = w[3:4]**2 * dt * I
    omega_i = w[-1:]**2 * dt * I

    #putting all the process noise into one matrix
    i = np.array([[V_1],
                  [theta_i],
                  [A_i],
                  [omega_i]])

    #Process noise covariance matrix Q | covariance of the of IMU + noise
    Q_i = np.array([[V_i, Z3, Z3,    Z3],
                    [Z3, theta_i, Z3,Z3],
                    [Z3, Z3, A_i,    Z3],
                    [Z3, Z3, Z3, omega_i]])

    #Rotated skew matrix for acceleration and gyro
    rsa = -R @ skew_symmetric(U[0:3] - x[10:13]) * dt
    rsg = R.T @ skew_symmetric(U[3:6] - x[13:16]) * dt

    #JACOBIAN MATRIX for how state evolves determinstically
    F_x = np.array([[I, I*dt, Z3, Z3, Z3,    Z3],
                    [Z3, I,  rsa, -R, Z3,  I*dt],
                    [Z3, Z3, rsg, Z3, -I*dt, Z3],
                    [Z3, Z3, Z3, I, Z3,      Z3],
                    [Z3, Z3, Z3, Z3, I,      Z3],
                    [Z3, Z3, Z3, Z3, Z3,      I]])

    #Error covariance update equation
    P = F_x @ P @ F_x.T + F_i @ Q_i @ F_i.T

    #LFG
    return x, P

def update_step(imu_data, x, P, V, dt, altimeter_data, gps_data):
    """
    UPDATE STEP FOR THE FILTER
    imu_data: [am, wm, magm, euler_angles]
    x: [p, v, q, ab, wb, g] state vector
    P: error covariance matrix
    V: measurement noise covariance matrix
    dt: time step
    """
    orientation = Quaternion(array=x[6:10])
    R = orientation.rotation_matrix

    y_a = imu_data[0:3]
    y_m = imu_data[6:9]
    y_z = altimeter_data
    y_p = gps_data
    
    b = np.array([[1], [0], [0]]) #magnetic field in global frame
    g = x[16:-1]                  #gravity vector from state

    if use_position():
        y_a_pred = R.T @ g
        y_m_pred = R.T @ b
        y_z_pred = x[2]
        y_p_pred = x[0:2]
        y_pred = np.hstack((y_a_pred, y_m_pred, y_z_pred, y_p_pred))
        y = np.hstack((y_a, y_m, y_z, y_p))
        dim = 9
    else:
        y_a_pred = R.T @ g
        y_m_pred = R.T @ b
        y_z_pred = x[2]
        y_pred = np.vstack((y_a_pred, y_m_pred, y_z_pred))
        y = np.vstack((y_a, y_m, y_z))
        dim  7
    
    g_cross = skew_symmetric(g)
    b_cross = skew_symmetric(b)
    q = x[6:10]
    three_d_theta = (imu_data[3:6]-x[13:16])*dt
    delta_q = q_rot(three_d_theta).elements

    H_y_a = -R(q).T @ g_cross @ delta_q
    H_y_m = 



    #K = P @ H.T @ (H @ P @ H.T + V).inv
    #P = (I - K @ H) @ P
    #return x, P

    