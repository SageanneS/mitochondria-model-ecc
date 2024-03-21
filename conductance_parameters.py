
def conductance_potential():

     # General Parameters
    C_m   = 0.58     # Membrane Capacitance
    F     = 96.485   # Faraday's Constant
    R     = 8.31441  # Gas Constant
    T     = 273+37.0 # Temperature (Kelvins)

    return C_m, F, R, T

def sarcolemma_conductances():

    g_K   = 21.6
    g_Kir = 3.7
    g_Na  = 268
    g_Cl  = 6.55
    g_NaK = 207.0*(10**-6)

    return g_K, g_Kir, g_Na, g_Cl, g_NaK

def tubular_conductances():

    g_K, g_Kir, g_Na, g_Cl, g_NaK = sarcolemma_conductances()

    nk      = 0.45
    nir     = 1.0
    nNa     = 0.1
    nCl     = 0.1
    npump   = 0.1   
    nCa     = 1.0  

    g_K_t   = nk*g_K
    g_Kir_t = nir*g_Kir
    g_Na_t  = nNa*g_Na
    g_Cl_t  = nCl*g_Cl
    g_NaK_t = npump*g_NaK
    g_Ca_t  = nCa*3.13

    return g_K_t, g_Kir_t, g_Na_t, g_Cl_t, g_NaK_t, g_Ca_t

