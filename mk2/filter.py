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
    return np.array([
        [q0, -q3, q2],
        [q3, q0, -q1],
        [-q2, q1, q0]
    ])

def q_rot(three_dimensional_theta):
    angle = norm(three_dimensional_theta)
    #this normlizes the axis. Tilts in the actual direction its going
    if angle > 0:
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
    #these are the measurements from the IMU, gps, and altimeter
    y_a = imu_data[0:3]
    y_m = imu_data[6:9]
    y_z = altimeter_data
    y_p = gps_data

    #magnetic field in global frame
    #gravity vector from state
    b = np.array([[1], [0], [0]])
    g = x[16:-1]

    #creates a matrix for predictions and sensor_data either with or without GPS
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
        dim = 7
    
    #gets the skew matrix of gravity and magnetic field
    g_cross = skew_symmetric(g)
    b_cross = skew_symmetric(b)
    q = x[6:10]

    #perform rotation
    orientation = Quaternion(array=x[6:10])
    R = orientation.rotation_matrix

    #jacobian matrix for measurement model
    H_a = -R(q).T @ g_cross
    H_m = -R(q).T @ b_cross
    H_ya_ma_wrt_q = np.vstack((H_a, H_m))

    H_x = np.zeros((dim, 16))
    H_x[0:6, 6:10] = H_ya_ma_wrt

    #=================#
    # I M P O R T A N T
    #=================#
    #IMPLEMENT ALTIMETER AND GPS MEASUREMENT JACOBIANS HERE LATER
    if use_position():
        H_x[6:8, 0:2] = np.eye(2) # sets the joacobian for position measurements x, y
        H_x[8:, 2:3] = 1          # sets the jacobian for altimeter z np.eye(1) = 1
    else:
        H[6:, 2:3] = 1      

    Q_x = 0.5 * np.vstack(( -q[1:], skew_symmetric(q))) # adds a stack to the top because its a 4x3 matrix hence -q[]

    #Jacobian true state with respect to error state
    X_x = np.zeros((15,16))
    X_x[6:10, 6:9] = Q_x 
    X_x[0:6, 0:6] = np.eye(6)
    X_x[10:, 9:] = np.eye(6)

    #state to sensor measurement jacobian
    H = H_x @ X_x

    #Kalman Gain -> high = more trust in measurements, low = more trust in prediction
    K = P @ H.T @ (H @ P @ H.T + V).inv
    #error gain computation 
    error_gain = K @ (y - y_pred)
    # update Quaternion with small angle approx.
    # q = q_old ⊗ δq
    x[6:10] = (Quaternion(array=x[6:10]) * q_rot(error_gain[6:9])).normalised.elements
    x[0:6] += error_gain[0:6]
    x[10:] += error_gain[9:] #not 10 because δtheta has 3 elements
    #Using joseph form to update P to ensure it remains symmetric positive definite
    P[:] = (np.eye(15)) - K @ H) @ P (np.eye(len(P)) - K  H).T + K@R@K.T
    #making a new jacobian G to account for quaternion update
    G = np.eye(15)
    # this resets the error theta part of the quaternion after update
    G[6:9, 6:9] -= skew_symmetric((1/2)*error_gain[6:9])
    P[:]= G @ P @ G.T
    return x, P

    