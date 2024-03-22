
def diffusion_constants():
    
    # Intercompartmental Diffusion
    tR       = 0.75                                                            # um^3/ms Intercompartment Ca diffusion parameter
    tSR      = tR                                                              # um^3/ms Intercompartment Ca diffusion parameter
    tCa      = tR                                                              # um^3/ms Intercompartment Ca diffusion parameter
    tATP     = 0.375                                                           # um^3/ms intercompartmental ATP diffusion parameter 
    tMg      = 1.5                                                             # um^3/ms intercompartmental Mg diffusion parameter
    
    return tR, tSR, tCa, tATP, tMg

def binding_sites():

    # Protein Binding Sites
    Cstot    = 31000.0                                                         # uM Total 
    Ttot     = 140.0                                                           # uM Total [T] Binding Sites 
    Ptot     = 1500.0                                                          # uM Total [P] Binding Sites

    return Cstot, Ttot, Ptot

def rate_constants_catrp():

    # Calcium Transport
    Le       = 0.00004                                                         # uM/ms*um^3 SR Ca leak constant
    kCson    = 0.000004                                                        # /uM*ms Rate of SR Ca binding from Calsequestrin
    kCsoff   = 0.005                                                           # /ms Rate of SR Ca dissociation from Calsequestrin
    vusr     = 2.4375                                                          # uM/ms* um^3 Rate constant of the SERCA pump
    Kcsr     = 0.27                                                            # uM Dissociation constant of Ca from SERCA
    n_S      = 1.7                                                             # SERCA Hill Coefficient 
    Kasr     = 0.02*799.6                                                      # ATP Dependence Lytton Scaled to IC of ATP
    kCatpon  = 0.15                                                            # /uM*ms Rate of Ca binding to ATP
    kCatpoff = 30.0                                                            # /ms Rate of Ca dissociation from ATP
    kMatpon  = 0.0015                                                          # /uMms Rate of Mg binding to ATP
    kMatpoff = 0.15                                                            # /ms Rate of Mg dissociation from ATP
    kTon     = 0.04425                                                         # /uMms Rate of Ca binding to Troponin
    kToff    = 0.115                                                           # /ms Rate of Ca dissociation from Troponin

    return Le, kCson, kCsoff, vusr, Kcsr, n_S, Kasr, kCatpon, kCatpoff, kMatpon, kMatpoff, kTon, kToff

def rate_constants_xbcyc():

    # Crossbridge Cycling
    k0on     = 0.0                                                             # /ms RU activation rate without two Ca bound
    k0off    = 0.15                                                            # /ms RU deactivation raplotte without two Ca bound
    kCaon    = 0.15                                                            # /ms RU activation rate with two Ca bound
    kCaoff   = 0.05                                                            # /ms RU deactivation rate with two Ca bound
    f0       = 0.5                                                             # /ms Rate of XB attachment
    fp       = 5.0                                                             # /ms Rate of pre-power stroke XB detachment
    h0       = 0.08                                                            # /ms Forward rate of the power stroke
    hp       = 0.06                                                            # /ms Reverse rate of power stroke
    g0       = 0.04                                                            # /ms Rate of post-power stroke XB detachment
    bbp      = 0.00000394                                                      # /ms Rate of myoplasmic phosphate degradation
    kp       = 0.00000362                                                      # um^3/ms Rate of transport of myoplasmic phosphate into the SR
    Ap       = 1.0                                                             # mM^2/ms Rate of phosphate precipitation
    Bp       = 0.0001                                                          # mM/ms Rate of phosphate precipitate solubilization
    PP       = 6.0                                                             # mM^2 phosphate solubility product

    return k0on, k0off, kCaon, kCaoff, f0, fp, h0, hp, g0, bbp, kp, Ap, Bp, PP
