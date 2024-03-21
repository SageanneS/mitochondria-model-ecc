# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:16:43 2019

@author: Sageanne Senneff
"""

"""
Created on Fri Apr 12 13:56:13 2019

This model describes the 2 compartment version (i.e. 1 Sarcolemma Compartment, 
1 Tubular Compartment) of the integrated skeletal muscle excitation-contraction
coupling model with mitochondrial dynamics for fast twitch fibers.

@author: Sageanne Senneff
"""
import time
start = time.time()

import scipy as sp
import numpy as np
import pylab as plt
from scipy.integrate import odeint
import scipy.io as io

from geometrical_cell_properties import muscle_fiber, half_sarcomere

class SingleFiber():
        
    # Define Global Variables
    global duration, dt
    global Vrp, urp
    global C_m, F, R, T, mr, tk, tCa
    global g_K, g_Kir, g_Na, g_Cl, g_NaK, g_K_t, g_Kir_t, g_Na_t, g_Cl_t, g_NaK_t, g_Ca_t
    global ko, ki, Nao, Nai, Clo, Cli, ko_t, ki_t, Nao_t, Nai_t, Clo_t, Cli_t 
    global E_Na, E_Cl, E_Na_t, E_Cl_t   
    global Gkir, I_nak, f_nak, Gkir_t, I_nak_t, f_nak_t
    global a_n, b_n, Van, Vbn, Vn, Vhk, Ahk, a_m, b_m, Vam, Vbm, Vm2, a_h, b_h, Vah, Vbh, Vh, Vs, As, Aa, a_fCa, K_fCa
    global sigma, Ks, Si, Kk, Kmk, KmNa
    global kL, kL_m, fallo, Vbar, K_RyR, alpha, i2, Cao_t, Le, vusr, tR, tSR, Kcsr, kPon, kPoff, Ptot, kMgon, kMgoff  
    global kCatpon, kCatpoff, kTon, kToff, kCson, kCsoff, Cstot, Ttot, PP, Ap, Bp, tATP, kMatpon, kMatpoff, tMg
    global k0on, k0off, kCaon, kCaoff, f0, hp, h0, fp, g0, bbp, kp
    global f_c, f_m, V_MCU, V_NCX, k_mPTP, V_ANT, V_F1F0, V_AGC, k_HYD, k_GLY, k_ETC, K_AGC, q_6, q_8, a1, a2, q_9, q_10
    global K_1, K_2, L, nA, q_7, Cam_thresh, p_3, KATP, theta, K_h, q_1, NADm_tot, q_2, q_3, q_4, q_5, p_4, C_p
    global alpha_c, alpham_m, Kank, n_S, Kasr

    # Simulation Parameters
    dt       = 0.001
    duration = 1000.0
    t        = sp.arange(0.0, duration, dt)
        
    # Resting Membrane Potential
    Vrp   = -78.1
    urp   = -78.1
    # External Current   
    def I_inj(self, t, i):
        I_inj = 0
        if (t >= 0 and t <= 0.6):
            I_inj = 200
        return I_inj             

    # General Parameters
    C_m   = 0.58                                                               # Membrane Capacitance
    F     = 96.485
    R     = 8.31441
    T     = 273+37.0

    # Muscle Fiber Geometry
    dx, r, Vsr, VsrS, p, ot, Gl, Ra, Ri, Vol, Ait, gl, b = muscle_fiber()
    # Half-Sarcomere Geometry
    Lx, RR, V0, VsrM, VsrC, V_t, V_m, Vm_t, Vm, Vsr_t, Vsr = half_sarcomere()

    # Ion Channel Conductances (Sarcolemma)
    g_K   = 21.6
    g_Kir = 3.7
    g_Na  = 268
    g_Cl  = 6.55
    g_NaK = 207.0*(10**-6)
    
    # Ion Channel Conductances (Tubular System)
    nk      = 0.45
    nir     = 1.0
    nNa     = 0.1
    nCl     = 0.1
    npump   = 0.1     
    g_K_t   = nk*g_K
    g_Kir_t = nir*g_Kir
    g_Na_t  = nNa*g_Na
    g_Cl_t  = nCl*g_Cl
    g_NaK_t = npump*g_NaK
    g_Ca_t  = 3.13
    
   # Ion Concentrations and Nernst Potentials (Sarcolemma)
    Nao   = 140.0
    Nai   = 10.0
    Clo   = 128.0
    Cli   = 5.7
    E_Na  = ((R*T)/F)*sp.log(Nao/Nai)
    E_Cl  = -((R*T)/F)*sp.log(Clo/Cli)
    
    # Ion Concentrations and Nernst Potentials (Tubular System)
    Nao_t   = Nao
    Nai_t   = Nai
    Clo_t   = Clo
    Cli_t   = Cli 
    E_Na_t  = ((R*T)/F)*sp.log(Nao_t/Nai_t)
    E_Cl_t  = -((R*T)/F)*sp.log(Clo_t/Cli_t)

    # Ion Channel Rate Parameters
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
    Kank  = 540.0                                                              # Hill Coefficient for ATP Dependence of NaK Pump

    # DHPR Gating Parameters
    K_fCa = 1.0
    a_fCa = 0.14                                                               # uM/ms 
    alpha = 0.2
    K_RyR = 4.5
    Vbar  = -20.0
    kL    = 0.002
    kL_m  = 1000.0
    fallo = 0.2
    i2    = 60.0

    # Calcium Transport and XB Dynamics Parameters
    Le       = 0.00004                                                         # uM/ms*um^3 SR Ca leak constant
    kCson    = 0.000004                                                        # /uM*ms Rate of SR Ca binding from Calsequestrin
    kCsoff   = 0.005                                                           # /ms Rate of SR Ca dissociation from Calsequestrin
    Cstot    = 31000.0                                                         # uM Total 
    vusr     = 2.4375                                                          # uM/ms* um^3 Rate constant of the SERCA pump
    Kcsr     = 0.27                                                            # uM Dissociation constant of Ca from SERCA
    n_S      = 1.7                                                             # SERCA Hill Coefficient 
    Kasr     = 0.02*799.6                                                      # ATP Dependence Lytton Scaled to IC of ATP
    kCatpon  = 0.15                                                            # /uM*ms Rate of Ca binding to ATP
    kCatpoff = 30.0                                                            # /ms Rate of Ca dissociation from ATP
    kMatpon  = 0.0015                                                          # /uMms Rate of Mg binding to ATP
    kMatpoff = 0.15                                                            # /ms Rate of Mg dissociation from ATP
    tR       = 0.75                                                            # um^3/ms Intercompartment Ca diffusion parameter
    tSR      = tR                                                              # um^3/ms Intercompartment Ca diffusion parameter
    tCa      = tR                                                              # um^3/ms Intercompartment Ca diffusion parameter
    tATP     = 0.375                                                           # um^3/ms intercompartmental ATP diffusion parameter 
    tMg      = 1.5                                                             # um^3/ms intercompartmental Mg diffusion parameter
    Ttot     = 140.0                                                           # uM Total [T] Binding Sites 
    kTon     = 0.04425                                                         # /uMms Rate of Ca binding to Troponin
    kToff    = 0.115                                                           # /ms Rate of Ca dissociation from Troponin
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
    Ptot     = 1500.0                                                          # uM Total [P] Binding Sites

    # Diffusion Time Constants
    tk = 559.0
    tCa = tR
    
    # General Mitochondria Parameters
    C_p      = 1.8                                                             # uM/mV Mitochondrial inner membrane capacitance divided by F
    a1       = 120.0                                                           # scaling factor between NADH consumption and change in membrane voltage
    a2       = 3.43                                                            # scaling factor between ATP production by ATPase and change in membrane voltage     
    f_c      = 1.0                                                             # Fraction of free over buffer-bound Ca in cytosol
    f_m      = 0.01                                                            # Fraction of free over buffer-bound Ca in mitochondria
    NADm_tot = 2970.0                                                          # uM Total concentration of mitochondrial pyridine nucleotide
    alpha_c  = 0.111
    alpham_m = 0.139
    
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
  
    # Mitochondrial Metabolism Parameters
    K_AGC = 0.14                                                               # uM Dissociation constant of Ca from AGC
    p_4   = 0.01                                                               # /mV Voltage dependence coefficient of AGC activity
    q_1   = 1.0                                                                # Michaelis-Menten-like constant for NAD+ consumption by the Krebs cycle
    q_2   = 0.1                                                                # uM S0.5 value for activation of the Krebs cycle by Ca
    V_AGC = 0.025                                                              # uM/ms Rate constant of NADH production via malate-aspartate shuttle
    k_GLY = 0.468                                                              # uM/ms Velocity of glycolysis

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

    # IKDR Channel
    def alpha_n(self,V):
        return a_n*(V-Vn)/(1-sp.exp(-(V-Vn)/Van))        
    def beta_n(self,V):
        return b_n*sp.exp(-(V-Vn)/Vbn)    
    def hkinf(self,V):
        return 1/(1+sp.exp((V-Vhk)/Ahk))
    def thk(self,V):
        return (sp.exp(-(V+40.0)/25.75))*(10**3) 
    # IKDRT Channel
    def alpha_n_t(self,u):
        return a_n*(u-Vn)/(1-sp.exp(-(u-Vn)/Van))        
    def beta_n_t(self,u):
        return b_n*sp.exp(-(u-Vn)/Vbn)   
    def hkinf_t(self,u):
        return 1.0/(1.0+sp.exp((u-Vhk)/Ahk))
    def thk_t(self,u):
        return (sp.exp(-(u+40.0)/25.75))*(10**3)  

    # IKIR Channel
    def Gkir(self,V, Ko, Ki):
        E_K   = ((R*T)/F)*sp.log(Ko/Ki)
        Kr    = Ko*sp.exp(-(sigma*E_K*F)/(R*T))
        ks    = Ks/((Si**2.0)*sp.exp(2.0*(1.0-sigma)*(V*F)/(R*T)))
        Kr1   = 1.0 + ((Kr**2.0)/Kk)
        y     = 1.0 - (1.0/(1.0 + (ks*Kr1)))
        gir   = g_Kir*(Kr**2.0)/(Kk+(Kr**2.0))
        return gir*y
    # IKIRT Channel
    def Gkir_t(self,u, Ko_t, Ki_t):
        E_K_t   = ((R*T)/F)*sp.log(Ko_t/Ki_t)
        Kr_t    = Ko_t*sp.exp(-(sigma*E_K_t*F)/(R*T))
        ks_t    = Ks/((Si**2.0)*sp.exp(2.0*(1.0-sigma)*(u*F)/(R*T)))
        Kr1_t   = 1.0 + ((Kr_t**2.0)/Kk)
        y_t     = 1.0 - (1.0/(1.0 + (ks_t*Kr1_t)))
        gir_t   = g_Kir_t*(Kr_t**2.0)/(Kk+(Kr_t**2.0))
        return gir_t*y_t

    # INA Channel
    def alpha_m(self, V):
        return a_m*(V-Vm2)/(1-sp.exp(-(V-Vm2)/Vam))
    def beta_m(self, V):
        return b_m*sp.exp(-(V-Vm2)/Vbm)
    def alpha_h(self, V):
        return a_h*sp.exp(-(V-Vh)/Vah)
    def beta_h(self, V):
        return b_h/(1+sp.exp(-(V-Vh)/Vbh))
    def Sinf(self,V):
        return 1.0/(1.0+sp.exp((V-Vs)/As))
    def ts(self,V):
        return (60.0/(0.2+(5.65*(((V+70.0)/100.0)**2.0))))*(10**3)
    # INAT Channel
    def alpha_m_t(self,u):
        return a_m*(u-Vm2)/(1-sp.exp(-(u-Vm2)/Vam))
    def beta_m_t(self,u):
        return b_m*sp.exp(-(u-Vm2)/Vbm)
    def alpha_h_t(self,u):
        return a_h*sp.exp(-(u-Vh)/Vah)
    def beta_h_t(self,u):
        return b_h/(1.0+sp.exp(-(u-Vh)/Vbh))
    def Sinf_t(self,u):
        return 1.0/(1.0+sp.exp((u-Vs)/As))
    def ts_t(self,u):
        return (60.0/(0.2+(5.65*(((u+70.0)/100.0)**2.0))))*(10**3)
    
    # NaK Pump
    def I_nak(self,V, Ko):
        return (F*g_NaK)/((1+(Kmk/(Ko**2)))*(1+((KmNa/Nai)**3)))
    def f_nak(self,V):
        o     = (sp.exp(Nao/67.3)-1)*(1.0/7.0)    
        return 1.0/(1.0 + (0.12*sp.exp(-(0.1*F*V)/(R*T))) + (0.04*o*sp.exp(-(F*V)/(R*T))))
    # NaKT Pump
    def I_nak_t(self,u, Ko_t):
        return (F*g_NaK_t)/((1+(Kmk/(Ko_t**2)))*(1.0+((KmNa/Nai_t)**3)))
    def f_nak_t(self,u):
        o_t   = (sp.exp(Nao_t/67.3)-1)*(1.0/7.0)    
        return 1.0/(1.0 + (0.12*sp.exp(-(0.1*F*u)/(R*T))) + (0.04*o_t*sp.exp(-(F*u)/(R*T))))
    
    # ICAT Channel
    def fCainf_t(self, Cai_t):
        return 1/(1 + (Cai_t/(K_fCa)))  
    def tfCa_t(self, Cai_t):
        return 1/(a_fCa*(1 + (Cai_t/K_fCa))) 
    def kC(self, u):
        return 0.5*alpha*sp.exp((u-Vbar)/(8*K_RyR))
    def kC_m(self, u):
        return 0.5*alpha*sp.exp(-(u-Vbar)/(8*K_RyR))

    # Ion Channels
    def I_KDR(self, V, n, hk, Ko, Ki):
        E_K = ((R*T)/F)*sp.log(Ko/Ki)
        return g_K  * n**4 * hk * (V - E_K) 
    def I_KDR_t(self, u, n_t, hk_t, Ko_t, Ki_t):
        E_K_t = ((R*T)/F)*sp.log(Ko_t/Ki_t)
        return g_K_t  * n_t**4 * hk_t * (u - E_K_t)  
    def I_KIR(self, V, Ko, Ki):
        E_K = ((R*T)/F)*sp.log(Ko/Ki)
        return Gkir(self,V, Ko, Ki) * (V - E_K) 
    def I_KIR_t(self, u, Ko_t, Ki_t):
        E_K_t = ((R*T)/F)*sp.log(Ko_t/Ki_t)
        return Gkir_t(self,u, Ko_t, Ki_t) * (u - E_K_t)
    def I_Na(self, V, m, h, S):
        return g_Na * m**3.0 * h * S * (V - E_Na)  
    def I_Na_t(self, u, m_t, h_t, S_t):
        return g_Na_t * m_t**3.0 * h_t * S_t * (u - E_Na_t)      
    def I_Cl(self, V):
        A = 1/(1+sp.exp((V-Aa)/Aa))
        return g_Cl * A**4.0 * (V - E_Cl)
    def I_Cl_t(self, u):
        A_t = 1/(1+sp.exp((u-Aa)/Aa))
        return g_Cl_t * A_t**4.0 * (u - E_Cl_t)    
    def I_NaK(self, V, Ko):     
        return I_nak(self,V, Ko)*f_nak(self,V)
    def I_NaK_t(self, u, ATP, Ko_t):     
        return I_nak_t(self,u, Ko_t)*f_nak_t(self,u)*(ATP/(Kank + ATP)) 
    def I_Ca_t(self, o_0, o_1, o_2, o_3, o_4, u, f_Ca, Cai_t, Cao_t):
        E_Ca_t = ((R*T)/F)*sp.log(Cao_t/Cai_t)
        return g_Ca_t*((o_0 + o_1 + o_2 + o_3 + o_4)*f_Ca)*(u - E_Ca_t)

    # Total Ionic Current
    def I(self, t, V, n, hk, m, h, S, Ko, Ki, i):
        return -self.I_inj(t, i) + self.I_KDR(V, n, hk, Ko, Ki) + self.I_KIR(V, Ko, Ki) + self.I_Na(V, m, h, S) + self.I_Cl(V) + self.I_NaK(V, Ko) 
    def I_t(self, u, n_t, hk_t, m_t, h_t, S_t, f_Ca, o_0, o_1, o_2, o_3, o_4, Cai_t, ATP, Ko_t, Ki_t, Cao_t):
        return self.I_KDR_t(u, n_t, hk_t, Ko_t, Ki_t) + self.I_KIR_t(u, Ko_t, Ki_t) + self.I_Na_t(u, m_t, h_t, S_t) + self.I_Cl_t(u) + self.I_NaK_t(u, ATP, Ko_t) + self.I_Ca_t(o_0, o_1, o_2, o_3, o_4, u, f_Ca, Cai_t, Cao_t)    
    def I_trans(self,V,u):
        return (V-u)/self.Ra

    # Calcium Dynamics (Myoplasm)
    def J_RyR(self, o_0, o_1, o_2, o_3, o_4, f_Ca, CaSR_t, Cai_t):            
        J_RYR = ((o_0 + o_1 + o_2 + o_3 + o_4)*f_Ca)*(CaSR_t - Cai_t)  
        return J_RYR 
    def J_SERCA_t(self, Cai_t, ATP_t):
        J_SERCA_t = vusr*((Cai_t**n_S)/(Kcsr + (Cai_t**n_S)))*(ATP_t/(Kasr + ATP_t))
        return J_SERCA_t
    def J_SERCA(self, Cai, ATP):
        J_SERCA = vusr*((Cai**n_S)/(Kcsr + (Cai**n_S)))*(ATP/(Kasr + ATP))  
        return J_SERCA
    
    # Calcium Dynamics (Mitochondria)    
    def J_MCU_t(self, Cai_t, Psi_t):
        return V_MCU*(((Cai_t/K_1)*((1 + (Cai_t/K_1))**3)*((2*F)*(Psi_t - 91.0)/(R*T)))/(((1 + (Cai_t/K_1))**4) + (L/((1 + (Cai_t/K_2))**nA))*(1 - sp.exp((-(2*F)*(Psi_t - 91.0))/(R*T)))))
    def J_MCU(self, Cai, Psi):
        return V_MCU*(((Cai/K_1)*((1 + (Cai/K_1))**3)*((2*F)*(Psi - 91.0)/(R*T)))/(((1 + (Cai/K_1))**4) + (L/((1 + (Cai/K_2))**nA))*(1 - sp.exp((-(2*F)*(Psi - 91.0))/(R*T)))))
    def J_NCX_t(self, Cai_t, Cam_t, Psi_t):
        return V_NCX*(sp.exp((0.5*F/(R*T))*(Psi_t - q_7))*sp.exp(sp.log(Cai_t/Cam_t)))/((1 + (9.4/Nai_t)**3)*(1 + (1.1/Cam_t)))
    def J_NCX(self, Cai, Cam, Psi):
        return V_NCX*(sp.exp((0.5*F/(R*T))*(Psi - q_7))*sp.exp(sp.log(Cai/Cam)))/((1 + (9.4/Nai_t)**3)*(1 + (1.1/Cam)))
    
    def J_mPTP_t(self, Cam_t, Cai_t, Psi_t, t):
        J_mPTP_t = (k_mPTP*(Cai_t - Cam_t)*sp.exp(p_3*Psi_t)) # Low Conductance
        if (Cam_t >= Cam_thresh).any():
            J_mPTP_t = (12.0/5.0)*J_mPTP_t                    # High Conductance
        return J_mPTP_t
    
    def J_mPTP(self, Cam, Cai, Psi, t):
        J_mPTP = (k_mPTP*(Cai - Cam)*sp.exp(p_3*Psi))                # Low Conductance
        if (Cam >= Cam_thresh).any():
            J_mPTP = (12.0/5.0)*(k_mPTP*(Cai - Cam)*sp.exp(p_3*Psi)) # High Conductance
        return J_mPTP 
    
    # Conservation Equations
    def T0(self, CaT, CaCaT, D0, D1, D2, A1, A2):
        return Ttot - CaT - CaCaT - D0 - D1 - D2 - A1 - A2
    def NADm_t(self, NADHm_t):
        return NADm_tot - NADHm_t
    def NADm(self, NADHm):
        return NADm_tot - NADHm
    
    # Mitochondrial Metabolism
    def J_PDH_t(self, NADHm_t, Cam_t):
        return k_GLY*(1/(q_1 + (NADHm_t/self.NADm_t(NADHm_t))))*(Cam_t/(q_2 + Cam_t))
    def J_PDH(self, NADHm, Cam):
        return k_GLY*(1/(q_1 + (NADHm/self.NADm(NADHm))))*(Cam/(q_2 + Cam))
    def J_AGC_t(self, Cai_t, Cam_t, Psi_t):
        return V_AGC*(Cai_t/(K_AGC + Cai_t))*(q_2/(q_2 + Cam_t))*(sp.exp(p_4*Psi_t))
    def J_AGC(self, Cai, Cam, Psi):
        return V_AGC*(Cai/(K_AGC + Cai))*(q_2/(q_2 + Cam))*(sp.exp(p_4*Psi))
    
    # Mitochondrial OXPHOS
    def J_ETC_t(self, NADHm_t, Psi_t):
        return k_ETC*(NADHm_t/(q_3 + NADHm_t))*(1/(1 + sp.exp((Psi_t - q_4)/q_5)))
    def J_ETC(self, NADHm, Psi):
        return k_ETC*(NADHm/(q_3 + NADHm))*(1/(1 + sp.exp((Psi - q_4)/q_5)))
    def J_F1F0_t(self, ATPm_t, Psi_t):
        return V_F1F0*(q_6/(q_6 + ATPm_t))*(1/(1 + sp.exp((q_7-Psi_t)/q_8)))
    def J_F1F0(self, ATPm, Psi):
        return V_F1F0*(q_6/(q_6 + ATPm))*(1/(1 + sp.exp((q_7-Psi)/q_8)))
    def J_ANT_t(self, ADP_t, ATP_t, Psi_t, ADPm_t, ATPm_t):
        return V_ANT*(ADP_t/(ADP_t + ATP_t*sp.exp(-(theta*F)*Psi_t/(R*T))) - ADPm_t/(ADPm_t + ATPm_t*sp.exp(((1-theta)*F)*Psi_t/(R*T))))*(1/(1 + KATP/ADP_t))
    def J_ANT(self, ADP, ATP, Psi, ADPm, ATPm):
        return V_ANT*(ADP/(ADP + ATP*sp.exp(-(theta*F)*Psi/(R*T))) - ADPm/(ADPm + ATPm*sp.exp(((1-theta)*F)*Psi/(R*T))))*(1/(1 + KATP/ADP))
    def J_Hleak_t(self, Psi_t):
        return q_9*Psi_t + q_10
    def J_Hleak(self, Psi):
        return q_9*Psi + q_10
    def J_HYD_t(self, ATP_t, Cai_t):
        return self.J_SERCA_t(Cai_t, ATP_t)/2 + k_HYD*(ATP_t/(ATP_t + K_h))
    def J_HYD(self, u, ATP, Cai, Ko_t):
        return self.I_NaK_t(u, ATP, Ko_t)/5 + self.J_SERCA(Cai, ATP)/2 + k_HYD*(ATP/(ATP + K_h))
    
    @staticmethod
    def dALLdt(X, t, self, i):
        """
        Integrate

        |  :param X:
        |  :param t:
        |  :return: calculate membrane potential & activation variables
        """
        
        V, n, hk, m, h, S, u, n_t, hk_t, m_t, h_t, S_t, c_0, o_0, c_1, o_1, c_2, o_2, c_3, o_3, c_4, o_4, f_Ca, Cai_t, Cai, CaSR_t, CaSR, CaCS_t, CaCS, CaATP_t, CaATP, MgATP_t, MgATP, Mg_t, Mg, ATP_t, ATP, ADP_t, ADP, CaT, CaCaT, D0, D1, D2, A1, A2, P, PSR, PCSR, Cam_t, Cam, NADHm_t, NADHm, ADPm_t, ADPm, ATPm_t, ATPm, Psi_t, Psi, Cao_t, Ko_t, Ki_t, Ko, Ki = X

        dVdt   = -(1.0/C_m) * (self.I(t, V, n, hk, m, h, S, Ko, Ki, i) + self.I_trans(V,u))    
        dndt   = self.alpha_n(V)*(1.0-n) - self.beta_n(V)*n
        dhkdt  = (self.hkinf(V) - hk)/self.thk(V)
        dmdt   = self.alpha_m(V)*(1.0-m) - self.beta_m(V)*m  
        dhdt   = self.alpha_h(V)*(1.0-h) - self.beta_h(V)*h       
        dSdt   = (self.Sinf(V) - S)/self.ts(V) 
        dudt = (1.0/C_m)*(((V -u)*((2*sp.pi*self.r*self.dx)/(self.Ra*self.Ait))) - self.I_t(u, n_t, hk_t, m_t, h_t, S_t, f_Ca, o_0, o_1, o_2, o_3, o_4, Cai_t, ATP, Ko_t, Ki_t, Cao_t))  

        dntdt  = self.alpha_n_t(u)*(1.0-n_t) - self.beta_n_t(u)*n_t
        dhktdt = (self.hkinf_t(u) - hk_t)/self.thk_t(u)
        dmtdt  = self.alpha_m_t(u)*(1.0-m_t) - self.beta_m_t(u)*m_t  
        dhtdt  = self.alpha_h_t(u)*(1.0-h_t) - self.beta_h_t(u)*h_t       
        dStdt  = (self.Sinf_t(u) - S_t)/self.ts_t(u) 
        dc0dt  = -kL*c_0 + kL_m*o_0 - 4*self.kC(u)*c_0 + self.kC_m(u)*c_1       
        do0dt  = kL*c_0 - kL_m*o_0 - (4*self.kC(u)/fallo)*o_0 + fallo*self.kC_m(u)*o_1
        dc1dt  = 4*self.kC(u)*c_0 - self.kC_m(u)*c_1 - (kL/fallo)*c_1 + kL_m*fallo*o_1 - 3*self.kC(u)*c_1 + 2*self.kC_m(u)*c_2
        do1dt  = (kL/fallo)*c_1 - (kL_m*fallo)*o_1 + (4*self.kC(u)/fallo)*o_0 - fallo*self.kC_m(u)*o_1 - (3*self.kC(u)/fallo)*o_1 + (2*fallo)*self.kC_m(u)*o_2
        dc2dt  = 3*self.kC(u)*c_1 - 2*self.kC_m(u)*c_2 - (kL/(fallo**2))*c_2 + (kL_m*(fallo**2))*o_2 - 2*self.kC(u)*c_2 + 3*self.kC_m(u)*c_3
        do2dt  = (3*self.kC(u)/fallo)*o_1 - (2*fallo)*self.kC_m(u)*o_2 + (kL/(fallo**2))*c_2 - (kL_m*(fallo**2))*o_2 - (2*self.kC(u)/fallo)*o_2 + (3*fallo)*self.kC_m(u)*o_3
        dc3dt  = 2*self.kC(u)*c_2 - 3*self.kC_m(u)*c_3 - (kL/(fallo**3))*c_3 + (kL_m*(fallo**3))*o_3 - self.kC(u)*c_3 + 4*self.kC_m(u)*c_4
        do3dt  = (kL/(fallo**3))*c_3 - (kL_m*(fallo**3))*o_3 + (2*self.kC(u)/fallo)*o_2 - (3*fallo)*self.kC_m(u)*o_3 - (self.kC(u)/fallo)*o_3 + (4*fallo)*self.kC_m(u)*o_4
        dc4dt  = self.kC(u)*c_3 - 4*self.kC_m(u)*c_4 - (kL/(fallo**4))*c_4 + (kL_m*(fallo**4))*o_4
        do4dt  = (self.kC(u)/fallo)*o_3 - (4*fallo)*self.kC_m(u)*o_4 + (kL/(fallo**4))*c_4 - (kL_m*(fallo**4))*o_4
        dfCadt = (self.fCainf_t(Cai_t) - f_Ca)/(self.tfCa_t(Cai_t))
        

        dCaitdt   = f_c*(-self.J_MCU_t(Cai_t, Psi_t)/self.V_t + self.J_NCX_t(Cai_t, Cam_t, Psi_t)/self.V_t - self.J_mPTP_t(Cam_t, Cai_t, Psi_t, t)/self.V_t) + self.I_Ca_t(o_0, o_1, o_2, o_3, o_4, u, f_Ca, Cai_t, Cao_t)/self.V_t + self.J_RyR(o_0, o_1, o_2, o_3, o_4, f_Ca, CaSR_t, Cai_t)/self.V_t + (Le*(CaSR_t - Cai_t))/self.V_t - self.J_SERCA_t(Cai_t, ATP_t)/self.V_t - (kCatpon*Cai_t)*ATP_t + kCatpoff*CaATP_t - tR*(Cai_t - Cai)/self.V_t
        dCaidt    = f_c*(-self.J_MCU(Cai, Psi)/self.V_m + self.J_NCX(Cai, Cam, Psi)/self.V_m - self.J_mPTP(Cam, Cai, Psi, t)/self.V_m) + (Le*(CaSR-Cai))/self.V_m - self.J_SERCA(Cai, ATP)/self.V_m + tR*(Cai_t - Cai)/self.V_m - (kCatpon*Cai)*ATP + kCatpoff*CaATP - kTon*Cai*self.T0(CaT, CaCaT, D0, D1, D2, A1, A2) + kToff*CaT - kTon*Cai*CaT + kToff*CaCaT - kTon*Cai*D0 + kToff*D1 - kTon*Cai*D1 + kToff*D2                        
        dCaSRtdt  = -self.J_RyR(o_0, o_1, o_2, o_3, o_4, f_Ca, CaSR_t, Cai_t)/self.Vsr_t + self.J_SERCA_t(Cai_t, ATP_t)/self.Vsr_t - Le*(CaSR_t - Cai_t)/self.Vsr_t - (kCson*CaSR_t)*(Cstot - CaCS_t) + kCsoff*CaCS_t - tSR*(CaSR_t - CaSR)/self.Vsr_t
        if PSR*0.001*CaSR >= PP: 
            dCaSRdt = self.J_SERCA(Cai, ATP)/self.Vsr - Le*(CaSR - Cai)/self.Vsr + tSR*(CaSR_t - CaSR)/self.Vsr - ((kCson*CaSR)*(Cstot - CaCS) - kCsoff*CaCS) - 1000*(Ap*(PSR*0.001*CaSR - PP)*0.001*PSR*CaSR)
        else:
            dCaSRdt = self.J_SERCA(Cai, ATP)/self.Vsr - Le*(CaSR - Cai)/self.Vsr + tSR*(CaSR_t - CaSR)/self.Vsr - ((kCson*CaSR)*(Cstot - CaCS) - kCsoff*CaCS) - 1000*(-Bp*PCSR*(PP -  PSR*0.001*CaSR))            
        dCaCStdt  = (kCson*CaSR_t)*(Cstot - CaCS_t) - kCsoff*CaCS_t
        dCaCSdt   = (kCson*CaSR)*(Cstot - CaCS) - kCsoff*CaCS
        dCaATPtdt = (kCatpon*Cai_t)*ATP_t - kCatpoff*CaATP_t - tATP*((CaATP_t - CaATP))/self.V_t
        dCaATPdt  = (kCatpon*Cai)*ATP - kCatpoff*CaATP + tATP*((CaATP_t - CaATP))/self.V_m
        dMgATPtdt = (kMatpon*Mg_t)*ATP_t - kMatpoff*MgATP_t - tATP*((MgATP_t - MgATP))/self.V_t
        dMgATPdt  = (kMatpon*Mg)*ATP - kMatpoff*MgATP + tATP*((MgATP_t - MgATP))/self.V_m
        dMgtdt    = -kMatpon*Mg_t*ATP_t + kMatpoff*MgATP_t - tMg*((Mg_t - Mg))/self.V_t
        dMgdt     = -kMatpon*Mg*ATP + kMatpoff*MgATP + tMg*((Mg_t - Mg))/self.V_m
        dATPtdt   = self.J_ANT_t(ADP_t, ATP_t, Psi_t, ADPm_t, ATPm_t)/self.V_t - self.J_HYD_t(ATP_t, Cai_t)/self.V_t - kCatpon*Cai_t*ATP_t + kCatpoff*CaATP_t - kMatpon*Mg_t*ATP_t + kMatpoff*MgATP_t - tATP*((ATP_t - ATP))/self.V_t
        dATPdt    = self.J_ANT(ADP, ATP, Psi, ADPm, ATPm)/self.V_m - self.J_HYD(u, ATP, Cai, Ko_t)/self.V_m - kCatpon*Cai*ATP + kCatpoff*CaATP - kMatpon*Mg*ATP + kMatpoff*MgATP + tATP*((ATP_t - ATP))/self.V_m              
        dADPtdt   = self.J_HYD_t(ATP_t, Cai_t)/self.V_t - (self.Vm_t/self.V_t)*self.J_ANT_t(ADP_t, ATP_t, Psi_t, ADPm_t, ATPm_t)/self.V_t - tATP*((ADP_t - ADP))/self.V_t
        dADPdt    = self.J_HYD(u, ATP, Cai, Ko_t)/self.V_m - (self.Vm/self.V_m)*self.J_ANT(ADP, ATP, Psi, ADPm, ATPm)/self.V_m + tATP*((ADP_t - ADP))/self.V_m  
        dCaTdt    = (kTon*Cai)*self.T0(CaT, CaCaT, D0, D1, D2, A1, A2) - kToff*CaT - (kTon*Cai)*CaT + kToff*CaCaT - k0on*CaT + k0off*D1           
        dCaCaTdt  = (kTon*Cai)*CaT - kToff*CaCaT - kCaon*CaCaT + kCaoff*D2   
        dD0dt     = (-kTon*Cai)*D0 + kToff*D1 + k0on*self.T0(CaT, CaCaT, D0, D1, D2, A1, A2) - k0off*D0
        dD1dt     = (kTon*Cai)*D0 - kToff*D1 + k0on*CaT - k0off*D1 - (kTon*Cai)*D1 + kToff*D2
        dD2dt     = (kTon*Cai)*D1 - kToff*D2 + kCaon*CaCaT - kCaoff*D2 - f0*D2 + fp*A1 + g0*A2  
        dA1dt     = f0*D2 - fp*A1 + hp*A2 - h0*A1
        dA2dt     = -hp*A2 + h0*A1 - g0*A2  
        dPdt      = 0.001*(self.J_HYD(u, ATP, Cai, Ko_t)/self.V_m) + 0.001*(h0*A1 - hp*A2) - bbp*P - kp*(P - PSR)/self.V_m   
        if PSR*0.001*CaSR >= PP:
            dPSRdt  = kp*(P - PSR)/self.Vsr - Ap*(PSR*0.001*CaSR - PP)*(0.001*PSR*CaSR)
            dPCSRdt = Ap*(PSR*0.001*CaSR - PP)*(0.001*PSR*CaSR)
        else:
            dPSRdt  = kp*(P - PSR)/self.Vsr + Bp*PCSR*(PP - PSR*0.001*CaSR)
            dPCSRdt = -Bp*PCSR*(PP - PSR*0.001*CaSR)
        
        dCamtdt   = f_m*(self.J_MCU_t(Cai_t, Psi_t)/self.Vm_t - self.J_NCX_t(Cai_t, Cam_t, Psi_t)/self.Vm_t + self.J_mPTP_t(Cam_t, Cai_t, Psi_t, t)/self.Vm_t)
        dCamdt    = f_m*(self.J_MCU(Cai, Psi)/self.Vm - self.J_NCX(Cai, Cam, Psi)/self.Vm + self.J_mPTP(Cam, Cai, Psi, t)/self.Vm)
        dNADHmtdt = self.J_PDH_t(NADHm_t, Cam_t)/self.Vm_t - self.J_ETC_t(NADHm_t, Psi_t)/self.Vm_t + self.J_AGC_t(Cai_t, Cam_t, Psi_t)/self.Vm_t
        dNADHmdt  = self.J_PDH(NADHm, Cam)/self.Vm - self.J_ETC(NADHm, Psi)/self.Vm + self.J_AGC(Cai, Cam, Psi)/self.Vm
        dADPmtdt  = self.J_ANT_t(ADP_t, ATP_t, Psi_t, ADPm_t, ATPm_t)/self.Vm_t - self.J_F1F0_t(ATPm_t, Psi_t)/self.Vm_t
        dADPmdt   = self.J_ANT(ADP, ATP, Psi, ADPm, ATPm)/self.Vm - self.J_F1F0(ATPm, Psi)/self.Vm
        dATPmtdt  = self.J_F1F0_t(ATPm_t, Psi_t)/self.Vm_t - self.J_ANT_t(ADP_t, ATP_t, Psi_t, ADPm_t, ATPm_t)/self.Vm_t
        dATPmdt   = self.J_F1F0(ATPm, Psi)/self.Vm - self.J_ANT(ADP, ATP, Psi, ADPm, ATPm)/self.Vm
        dPsitdt   = (a1*self.J_ETC_t(NADHm_t, Psi_t) - a2*self.J_F1F0_t(ATPm_t, Psi_t) - self.J_ANT_t(ADP_t, ATP_t, Psi_t, ADPm_t, ATPm_t) - self.J_Hleak_t(Psi_t) - self.J_NCX_t(Cai_t, Cam_t, Psi_t) - 2*self.J_MCU_t(Cai_t, Psi_t) - 2*self.J_mPTP_t(Cam_t, Cai_t, Psi_t, t) - self.J_AGC_t(Cai_t, Cam_t, Psi_t))/C_p
        dPsidt    = (a1*self.J_ETC(NADHm, Psi) - a2*self.J_F1F0(ATPm, Psi) - self.J_ANT(ADP, ATP, Psi, ADPm, ATPm) - self.J_Hleak(Psi) - self.J_NCX(Cai, Cam, Psi) - 2*self.J_MCU(Cai, Psi) - 2*self.J_mPTP(Cam, Cai, Psi, t) - self.J_AGC(Cai, Cam, Psi))/C_p
        
        dCaotdt = (self.I_Ca_t(o_0, o_1, o_2, o_3, o_4, u, f_Ca, Cai_t, Cao_t)/1000*F*1000*self.Vsr) - (Cao_t - Cao_t)/tCa
        dKotdt  = (((self.I_KIR_t(u, Ko_t, Ki_t) + self.I_KDR_t(u, n_t, hk_t, Ko_t, Ki_t) -(2*self.I_NaK_t(u, ATP, Ko_t)))/1000*F*1000*self.Vsr) - ((Ko_t - Ko)/tk) - (Ko_t - Ko_t)/tk)
        dKitdt  = dKotdt*(15.5/25.0)
        dKodt   = (((self.I_KIR(V, Ko, Ki) + self.I_KDR(V, n, hk, Ko, Ki) - 2*self.I_NaK(u, Ko))/1000*F*1000*self.VsrS)-((Ko - Ko)/tk + (Ko - Ko)/tk + (Ko - Ko_t)/tk))
        dKidt   = dKodt*(15.5/25.0)

        return dVdt, dndt, dhkdt, dmdt, dhdt, dSdt, dudt, dntdt, dhktdt, dmtdt, dhtdt, dStdt, dc0dt, do0dt, dc1dt, do1dt, dc2dt, do2dt, dc3dt, do3dt, dc4dt, do4dt, dfCadt, dCaitdt, dCaidt, dCaSRtdt, dCaSRdt, dCaCStdt, dCaCSdt, dCaATPtdt, dCaATPdt, dMgATPtdt, dMgATPdt, dMgtdt, dMgdt, dATPtdt, dATPdt, dADPtdt, dADPdt, dCaTdt, dCaCaTdt, dD0dt, dD1dt, dD2dt, dA1dt, dA2dt, dPdt, dPSRdt, dPCSRdt, dCamtdt, dCamdt, dNADHmtdt, dNADHmdt, dADPmtdt, dADPmdt, dATPmtdt, dATPmdt, dPsitdt, dPsidt, dCaotdt, dKotdt, dKitdt, dKodt, dKidt

    def Main(self):
        
        V_0    = Vrp
        n_0    = (a_n*(V_0-Vn)/(1-sp.exp(-(V_0-Vn)/Van)))/((a_n*(V_0-Vn)/(1-sp.exp(-(V_0-Vn)/Van))) + (b_n*sp.exp(-(V_0-Vn)/Vbn)))
        hk_0   = 1/(1+sp.exp((V_0-Vhk)/Ahk))
        m_0    = (a_m*(V_0-Vm2)/(1-sp.exp(-(V_0-Vm2)/Vam)))/((a_m*(V_0-Vm2)/(1-sp.exp(-(V_0-Vm2)/Vam))) + (b_m*sp.exp(-(V_0-Vm2)/Vbm)))
        h_0    = (a_h*sp.exp(-(V_0-Vh)/Vah))/((a_h*sp.exp(-(V_0-Vh)/Vah)) + (b_h/(1+sp.exp(-(V_0-Vh)/Vbh))))
        S_0    = 1.0/(1.0+sp.exp((V_0-Vs)/As))

        u_0    = urp
        n_t_0  = (a_n*(u_0-Vn)/(1-sp.exp(-(u_0-Vn)/Van)))/((a_n*(u_0-Vn)/(1-sp.exp(-(u_0-Vn)/Van))) + (b_n*sp.exp(-(u_0-Vn)/Vbn)))
        hk_t_0 = 1/(1+sp.exp((u_0-Vhk)/Ahk))
        m_t_0  = (a_m*(u_0-Vm2)/(1-sp.exp(-(u_0-Vm2)/Vam)))/((a_m*(u_0-Vm2)/(1-sp.exp(-(u_0-Vm2)/Vam))) + (b_m*sp.exp(-(u_0-Vm2)/Vbm)))
        h_t_0  = (a_h*sp.exp(-(u_0-Vh)/Vah))/((a_h*sp.exp(-(u_0-Vh)/Vah)) + (b_h/(1+sp.exp(-(u_0-Vh)/Vbh))))
        S_t_0  = 1.0/(1.0+sp.exp((u_0-Vs)/As))
        c_0_0  = 1.0
        o_0_0  = 0.0
        c_1_0  = 0.0
        o_1_0  = 0.0
        c_2_0  = 0.0
        o_2_0  = 0.0
        c_3_0  = 0.0
        o_3_0  = 0.0
        c_4_0  = 0.0
        o_4_0  = 0.0
        f_Ca_0 = 1/(1 + (a_fCa/K_fCa))

        Cai_t_0   = 0.1
        Cai_0     = 0.1
        CaSR_t_0  = 1500.0
        CaSR_0    = 1500.0
        CaCS_t_0  = 16900.0
        CaCS_0    = 16900.0
        CaATP_t_0 = 0.4
        CaATP_0   = 0.4
        MgATP_t_0 = 7200.0
        MgATP_0   = 7200.0
        Mg_t_0    = 1000.0
        Mg_0      = 1000.0
        ATP_t_0   = 799.6
        ATP_0     = 799.6
        ADP_t_0   = 2.92
        ADP_0     = 2.92
        CaT_0     = 25.0
        CaCaT_0   = 3.0
        D0_0      = 0.8   
        D1_0      = 1.2
        D2_0      = 3.0
        A1_0      = 0.23
        A2_0      = 0.23
        P_0       = 0.23
        PSR_0     = 0.23 
        PCSR_0    = 0.23 

        Cam_t_0   = 0.1
        Cam_0     = 0.1
        NADHm_t_0 = 560.4                                                           
        NADHm_0   = 560.4                                                            
        ADPm_t_0  = 36.4                                                              
        ADPm_0    = 36.4                                                             
        ATPm_t_0  = 9963.6                                                            
        ATPm_0    = 9963.6                                                            
        Psi_t_0   = 190.0                                                           
        Psi_0     = 190.0   

        Cao_t_0   = 1300.0
        Ko_t_0    = 4.0
        Ki_t_0    = 154.5
        Ko_0      = 4.0
        Ki_0      = 154.5
    
        for i in range(1):
 
            X    = odeint(self.dALLdt, [V_0, n_0, hk_0, m_0, h_0, S_0, u_0, n_t_0, hk_t_0, m_t_0, h_t_0, S_t_0, c_0_0, o_0_0, c_1_0, o_1_0, c_2_0, o_2_0, c_3_0, o_3_0, c_4_0, o_4_0, f_Ca_0, Cai_t_0, Cai_0, CaSR_t_0, CaSR_0, CaCS_t_0, CaCS_0, CaATP_t_0, CaATP_0, MgATP_t_0, MgATP_0, Mg_t_0, Mg_0, ATP_t_0, ATP_0, ADP_t_0, ADP_0, CaT_0, CaCaT_0, D0_0, D1_0, D2_0, A1_0, A2_0, P_0, PSR_0, PCSR_0, Cam_t_0, Cam_0, NADHm_t_0, NADHm_0, ADPm_t_0, ADPm_0, ATPm_t_0, ATPm_0, Psi_t_0, Psi_0, Cao_t_0, Ko_t_0, Ki_t_0, Ko_0, Ki_0], self.t, args=(self,i))
            
            V    = X[:,0]
            n    = X[:,1]
            hk   = X[:,2]
            m    = X[:,3]
            h    = X[:,4]
            S    = X[:,5]
            u    = X[:,6]

            n_t  = X[:,7]
            hk_t = X[:,8]
            m_t  = X[:,9]
            h_t  = X[:,10]
            S_t  = X[:,11]
            c_0  = X[:,12]
            o_0  = X[:,13]
            c_1  = X[:,14]
            o_1  = X[:,15]
            c_2  = X[:,16]
            o_2  = X[:,17]
            c_3  = X[:,18]
            o_3  = X[:,19]
            c_4  = X[:,20]
            o_4  = X[:,21]
            f_Ca = X[:,22]
            
            Cai_t   = X[:,23]
            Cai     = X[:,24]
            CaSR_t  = X[:,25]
            CaSR    = X[:,26]
            CaCS_t  = X[:,27]
            CaCS    = X[:,28]
            CaATP_t = X[:,29]
            CaATP   = X[:,30]
            MgATP_t = X[:,31]
            MgATP   = X[:,32]
            Mg_t    = X[:,33]
            Mg      = X[:,34]
            ATP_t   = X[:,35]
            ATP     = X[:,36]
            ADP_t   = X[:,37]
            ADP     = X[:,38]
            CaT     = X[:,39]
            CaCaT   = X[:,40]
            D0      = X[:,41]   
            D1      = X[:,42]
            D2      = X[:,43]
            A1      = X[:,44]
            A2      = X[:,45]
            P       = X[:,46]
            PSR     = X[:,47] 
            PCSR    = X[:,48] 
            
            Cam_t   = X[:,49]
            Cam     = X[:,50]
            NADHm_t = X[:,51]                                                           
            NADHm   = X[:,52]                                                            
            ADPm_t  = X[:,53]                                                             
            ADPm    = X[:,54]                                                             
            ATPm_t  = X[:,55]                                                            
            ATPm    = X[:,56]                                                            
            Psi_t   = X[:,57]                                                              
            Psi     = X[:,58]                                                                
            
            Cao_t   = X[:,59]
            Ko_t    = X[:,60]
            Ki_t    = X[:,61]
            Ko      = X[:,62]
            Ki      = X[:,63]

            IKDR   = self.I_KDR(V, n, hk, Ko, Ki)
            IKIR   = self.I_KIR(V, Ko, Ki)
            INA    = self.I_Na(V, m, h, S)
            ICL    = self.I_Cl(V)
            INAK   = self.I_NaK(V, Ko)
            IKDR_t = self.I_KDR_t(u, n_t, hk_t, Ko_t, Ki_t)
            IKIR_t = self.I_KIR_t(u, Ko_t, Ki_t)
            INA_t  = self.I_Na_t(u, m_t, h_t, S_t)
            ICL_t  = self.I_Cl_t(u)
            INAK_t = self.I_NaK_t(u, ATP, Ko_t)    
            ICA_t  = self.I_Ca_t(o_0, o_1, o_2, o_3, o_4, u, f_Ca, Cai_t, Cao_t)
            ITRANS = self.I_trans(V,u)
            JRYR    = self.J_RyR(o_0, o_1, o_2, o_3, o_4, f_Ca, CaSR_t, Cai_t)
            JSERCAT = self.J_SERCA_t(Cai_t, ATP_t)
            JSERCA = self.J_SERCA(Cai, ATP)
            JMCUT  = self.J_MCU_t(Cai_t, Psi_t)
            JMCU   = self.J_MCU(Cai, Psi)
            JNCXT  = self.J_NCX_t(Cai_t, Cam_t, Psi_t)
            JNCX   = self.J_NCX(Cai, Cam, Psi)
            JMPTPT = self.J_mPTP_t(Cam_t, Cai_t, Psi_t, self.t)
            JMPTP  = self.J_mPTP(Cam, Cai, Psi, self.t)
            JANTT  = self.J_ANT(ADP_t, ATP_t, Psi_t, ADPm_t, ATPm_t)
            JANT   = self.J_ANT(ADP, ATP, Psi, ADPm, ATPm)
            JF1F0T = self.J_F1F0_t(ATPm_t, Psi_t)
            JF1F0  = self.J_F1F0(ATPm, Psi)
            JETCT  = self.J_ETC_t(NADHm_t, Psi_t)
            JETC   = self.J_ETC(NADHm, Psi)
            JPDHT  = self.J_PDH_t(NADHm_t, Cam_t)
            JPDH   = self.J_PDH(NADHm, Cam)
            JAGCT  = self.J_AGC_t(Cai_t, Cam_t, Psi_t)
            JAGC   = self.J_AGC(Cai, Cam, Psi)
            NADM_t = self.NADm_t(NADHm_t)
            NADM   = self.NADm(NADHm)
            JHleak_t = self.J_Hleak_t(Psi_t)
            JHleak = self.J_Hleak(Psi)
            JHYD_t = self.J_HYD_t(ATP_t, Cai_t)
            JHYD = self.J_HYD(u, ATP, Cai, Ko_t) 

        io.savemat('StateVariables_Fluxes.mat', mdict={'X': X, 'Flux' : (ICA_t, JRYR, JSERCAT, JSERCA, JMCUT, JMCU, JNCXT, JNCX, JMPTPT, JMPTP, JANTT, JANT, JF1F0T, JF1F0)})

if __name__ == '__main__':
    runner = SingleFiber()
    runner.Main()
    
end = time.time()
print(end - start)
