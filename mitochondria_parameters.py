
def mito_properties():

    # General Mitochondria Parameters
    C_p      = 1.8                                                             # uM/mV Mitochondrial inner membrane capacitance divided by F
    a1m       = 120.0                                                          # scaling factor between NADH consumption and change in membrane voltage
    a2m       = 3.43                                                           # scaling factor between ATP production by ATPase and change in membrane voltage     
    f_c      = 1.0                                                             # Fraction of free over buffer-bound Ca in cytosol
    f_m      = 0.01                                                            # Fraction of free over buffer-bound Ca in mitochondria
    NADm_tot = 2970.0                                                          # uM Total concentration of mitochondrial pyridine nucleotide
    alpha_c  = 0.111
    alpham_m = 0.139

    return C_p, a1m, a2m, f_c, f_m, NADm_tot, alpha_c, alpham_m


def calcium_exchange(): 

    # Mitochondrial Calcium Exchange Parameters
    p_1        = 0.1                                                           # /mV Voltage dependence coefficient of MCU activity
    p_3        = 0.075                                                         # /mV MODEL FIT Voltage dependence coefficient of calcium leak
    nA         = 2.8                                                           # cooperativity paramter for MCU
    L          = 130.0                                                         # Allosteric equilibrium constant for uniporter conformations
    K_1        = 19.0                                                          # uM Dissociation constant for Ca translocation by MCU
    K_2        = 0.38                                                          # uM Dissociation constant for MCU activation by Ca
    V_MCU      = 0.0215                                                        # uM/ms max uptake rate of MCU                                                                
    k_mPTP     = 0.000008                                                      # /ms Rate constant of bidirectional Ca leak from mitochondria
    V_NCX      = 0.00035                                                       # uM/ms max uptake of NCX                                          
    Cam_thresh = 2328.0                                                        # uM mPTP Ca Threshold Value 

    return p_1, p_3, nA, L, K_1, K_2, V_MCU, k_mPTP, V_NCX, Cam_thresh

def metabolism():

    # Mitochondrial Metabolism Parameters
    K_AGC = 0.14                                                               # uM Dissociation constant of Ca from AGC
    p_4   = 0.01                                                               # /mV Voltage dependence coefficient of AGC activity
    q_1   = 1.0                                                                # Michaelis-Menten-like constant for NAD+ consumption by the Krebs cycle
    q_2   = 0.1                                                                # uM S0.5 value for activation of the Krebs cycle by Ca
    V_AGC = 0.025                                                              # uM/ms Rate constant of NADH production via malate-aspartate shuttle
    k_GLY = 0.468                                                              # uM/ms Velocity of glycolysis

    return K_AGC, p_4, q_1, q_2, V_AGC, k_GLY

def OXPHOS():

    # Mitochondrial OXPHOS Parameters
    q_3     = 100.0                                                            # uM Michaelis-Menten constant for NADH consumption by the ETC
    q_4     = 177.0                                                            # mV Voltage dependence coefficient 1 of ETC activity
    q_5     = 5.0                                                              # mV Voltage dependence coefficient 2 of ETC activity
    q_6     = 10000.0                                                          # uM Inhibition constant of ATPase activity by ATP
    q_7     = 190.0                                                            # mV Voltage dependence coefficient of ATPase activity
    q_8     = 8.5                                                              # mV Voltage dependence coefficient of ATPase activity
    q_9     = 0.0020                                                           # uM/ms*mV Voltage dependence of the proton leak
    q_10    = -0.030                                                           # uM/ms Rate constant of the voltage-independent proton leak
    K_h     = 150.0                                                            # uM Michaelis-Menten constant for ATP hydrolysis
    k_HYD   = 0.0114417                                                        # uM/ms Maximal rate of ATP hydrolysis
    V_ANT   = 8.123                                                            # uM/ms Rate constant of the adenine nucleotide translocator
    V_F1F0  = 3.6                                                              # uM/ms Rate constant of the F1FO ATPase
    KATP    = 3.5                                                              # uM
    theta   = 0.35                                                             # ANT parameter
    V_AT    = 0.50915                                                          # /ms rate of ATP transport from mitochondria into myoplasm 
    k_ETC   = 0.764                                                            # uM/ms Rate constant of NADH oxidation by ETC

    return q_3, q_4, q_5, q_6, q_7, q_8, q_9, q_10, K_h, k_HYD, V_ANT, V_F1F0, KATP, theta, V_AT, k_ETC
