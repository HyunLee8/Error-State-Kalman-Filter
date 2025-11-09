import numpy as np

uncertainty_p = 5.0
uncertainty_v = 4.0

def filter(Xk1, Pk1, delta_t, a):
    
    #Kinematic matrix
    Fk = np.array([[1, delta_t],
                  [0, 1]])
    Fkt = np.transpose(Fk)

    #now we have our estimates for Xk and Pk
    Xk = Fk @ Xk1
    Pk = Fk @ Pk1 @ Fkt

    #Now lets get External Influences
    #position = position0 + delta_t*velocity0 + 1/2*a*delta_t^2
    #velocity =                     velocity0 + a*delta_t
    # We need Bk to be the factor that multiplies the acceleration 'a'

    #control matrix
    Bk = np.array([[(delta_t**2)/2],
                   [delta_t]])
    #control vector 
    Uk = a

    #EXTERNAL UNCERTAINTY
    uncertainty_p_ext = 2.0
    uncertainty_v_ext = 1.5
    varP_ext = uncertainty_p_ext**2
    varV_ext= uncertainty_v_ext**2
    rho_ext = 0.2
    covar_ext = rho_ext * np.sqrt(varP_ext) * np.sqrt(varV_ext)
    covariance_matrix_ext = np.array([[varP_ext, covar_ext],
                                      [covar_ext, varV_ext]])  

    Qk = covariance_matrix_ext

    Xk = Fk @ Xk1 + Bk*Uk
    Pk = Fk @ Pk1 @ Fkt + Qk

    return Xk, Pk


while True:
    count = 0

    #Here this is just a placeholder for real data
    position = float(input("Enter position: "))
    velocity = float(input("Enter velocity: "))
    a = float(input("Enter acceleration: "))
    delta_t = 1

    if count == 0:
        Xk = np.array([[position],
                       [velocity]])
        varP = uncertainty_p**2
        varV = uncertainty_v**2
        rho = 0.8
        covar = rho * np.sqrt(varP) * np.sqrt(varV)
        Pk = np.array([[varP, covar],
                       [covar, varV]])
        count += 1

    Xk, Pk = filter(Xk1=Xk, Pk1=Pk, delta_t=delta_t, a=a)
    print("Estimated State Xk:\n", Xk)
    print("Estimated Covariance Pk:\n", Pk)
    
    





