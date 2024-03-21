

def voltage_channel_rates():

    a_n   = 0.0131
    Van   = 7.0
    Vn    = -40.0
    b_n   = 0.067
    Vbn   = 40.0
    Vhk   = -40.0
    Ahk   = 7.5
    Kk    = 950.0
    Ks    = 1.0
    Si    = 10.0
    sigma = 0.4
    a_m   = 0.288
    Vm2   = -46.0
    Vam   = 10.0
    b_m   = 1.38
    Vbm   = 18.0
    a_h   = 0.0081
    Vh    = -45.0
    Vah   = 14.7
    b_h   = 4.38
    Vbh   = 9.0
    Vs    = -68.0
    As    = 7.1
    Va    = 70.0
    Aa    = 150.0
    Kmk   = 1.0
    KmNa  = 13.0
    Kank  = 540.0   

    return a_n, Van, Vn, b_n, Vbn, Vhk, Ahk, Kk, Ks, Si, sigma, a_m, Vm2, Vam, b_m, Vbm, a_h, Vh, Vah, b_h, Vbh, Vs, As, Va, Aa, Kmk, KmNa, Kank

def voltage_calcium_channel_rates():

    # DHPR Gating Parameters
    K_fCa = 1.0
    a_fCa = 0.14                                                               
    alpha = 0.2
    K_RyR = 4.5
    Vbar  = -20.0
    kL    = 0.002
    kL_m  = 1000.0
    fallo = 0.2
    i2    = 60.0

    return K_fCa, a_fCa, alpha, K_RyR, Vbar, kL, kL_m, fallo, i2
