import scipy as sp

def muscle_fiber():

    # Muscle Fiber Geometry
    dx    = 100.0*(10**-4)
    r     = 20.0*(10**-4)
    Vsr   = 1.0*(10**-6)
    VsrS  = 4.1*(10**-6)                                                       # volume-surface ratio sarcolemma
    p     = 0.003
    ot    = 0.34
    Gl    = 3.7*p*ot
    Ra    = 0.150
    Ri    = 0.125
    Vol   = sp.pi*dx*(r**2)
    Ait   = p*Vol/Vsr
    gl    = (2.0*sp.pi*r*dx*Gl)/(r/20.0)
    b     = gl/Ait

    return dx, r, Vsr, VsrS, p, ot, Gl, Ra, Ri, Vol, Ait, gl, b

def half_sarcomere():

    # Half-Sarcomere Geometry
    Lx     = 1.1                                                               # um sarcomere length (z-line to m-line)
    RR     = 0.5                                                               # um myofibre radius
    V0     = 0.791*(Lx*sp.pi*(RR**2))                                          # um^3 Total myoplasmic volume
    VsrM   = 0.159*(Lx*sp.pi*(RR**2))                                          # um^3 Total mitochondrial myoplasmic volume
    VsrC   = 0.05*(Lx*sp.pi*(RR**2))                                           # um^3 SR Volume
    V_t    = 0.01*V0    #V_TM                                                  # um^3 TSR Volume     
    V_m    = 0.99*V0    #V_M                                                   # um^3 Myoplasm less TSR less Mitochondria Volume
    Vm_t   = 0.01*VsrM  #V_TMC                                                 # um^3 TSR Volume     
    Vm     = 0.99*VsrM  #V_MC 
    Vsr_t  = 0.01*VsrC  #V_TSR                                                 # um^3 TSR SR Volume 
    Vsr    = 0.99*VsrC  #V_SR                                                  # um^3 SR less TSR Volume
    
    return Lx, RR, V0, VsrM, VsrC, V_t, V_m, Vm_t, Vm, Vsr_t, Vsr

