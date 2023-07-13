from pickle import TRUE
import string
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as up
import uncertainties as un
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from generatetable import generatetable
from generatetable import round_to_significant

data = np.loadtxt('data.txt')

hours = data[0:, 0] #stunden
min = data[0:, 1] #minuten
sec = data[0:, 2] #sekunden
R_sample = data[0:, 3]# in ohm
R_shield = data[0:, 4]# in ohm
U = data[0:, 5] # in V
I = data[0:, 6]*10**-3# in A


# Umrechnen in Sekunden
def time_in_sec(h,min,s):
    return 60*60*h+60*min+s

t= time_in_sec(hours,min,sec)# in s

# Umrechnen Widerstände in Temperatur
def resistance_to_temperature(R):
    return 0.00134*R**2+2.296*R - 243.02

T_sample= resistance_to_temperature(R_sample)# celsius
T_shield= resistance_to_temperature(R_shield)# celsius

# Umrechnen von Celsius in Kelvin
def celsius_to_kelvin(T):
    return T+273.15

T_sample_kelvin= celsius_to_kelvin(T_sample)# kelvin
T_shield_kelvin= celsius_to_kelvin(T_shield)# kelvin


######################C_p########################
# Calculate Heat added to the sample

dt = np.diff(t)# differences between elements in array
dt = np.insert(dt, 0, 0)

def H_calc(I,dt,U): 
    return I * dt * U

H = H_calc(I,dt,U) # in joule

# Calculate C_p
M = 63.55 # g/mol
m = 342 # mass[g]
dT_sample_kelvin = np.diff(T_sample_kelvin)
dT_sample_kelvin = np.insert(dT_sample_kelvin, 0, 0)

def C_p_calc(M,H,m,dT):
    return (M*H)/(m*dT)

C_p= C_p_calc(M,H,m,dT_sample_kelvin) # J/mol K

for e0,e1,e2,e3 in zip(R_sample,T_sample_kelvin,C_p,t):
    print("R= ", e0,"T= ", e1, "C_p= ", e2, "t= ", e3)
######################C_V########################
for e1,e2 in zip(dt,H):
    print("dt= ", e1, "H= ", e2)
#determine alpha(T)
alpha_data = np.loadtxt('alpha_data.txt')

alpha_T = alpha_data[1:, 0] # in K
alpha_alpha = alpha_data[1:, 1]*10**-6 # in grd^-1

plt.plot(alpha_T, alpha_alpha, '.k', label="Data")

def linfunc(x,m,c):
    return m*x+c

x=np.linspace(0,310,100)
pars, covs = curve_fit(linfunc, xdata=alpha_T,ydata=alpha_alpha)
plt.plot(x, linfunc(x,*pars), '-b', label=r"Linear regression")

alpha_m = ufloat(pars[0], np.sqrt(covs[0, 0]))
alpha_c = ufloat(pars[1], np.sqrt(covs[1, 1]))
print("m =","{:.{}f}".format(alpha_m, 10), "1/grd K")
print("c =","{:.{}f}".format(alpha_c, 10), "1/grd")

plt.xlim(0,310)
plt.xlabel(r'$T\,/\,\mathrm{K}$')
plt.ylabel(r'$\alpha\,/\,\mathrm{\frac{1}{grd}}$')

plt.grid("True")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("build/alpha_plot.pdf")
#plt.show()

alpha = np.ones(len(R_sample))
#def alpha_func(T):
#    for i in range(len(T)):
#        j = 0
#        while(T[i]>alpha_T[j]):
#            j = j+1
#        alpha[i] = (alpha_alpha[j]-alpha_alpha[j-1])*(T[i]-alpha_T[j-1])/(alpha_T[j]-alpha_T[j-1])+alpha_alpha[j-1]
#    return alpha

def alpha_func(T):
    alpha = np.ones(len(T))
    for i in range(len(T)):
        j = 0
        if T[i] < alpha_T[0]:  # Handle the case when T[i] is smaller than the first threshold
            alpha[i] = alpha_alpha[0]
        elif T[i] >= alpha_T[-1]:  # Handle the case when T[i] is larger than or equal to the last threshold
            alpha[i] = alpha_alpha[-1]
        else:
            while j < len(alpha_T) - 1 and T[i] > alpha_T[j]:
                j += 1
            alpha[i] = (alpha_alpha[j] - alpha_alpha[j-1]) * (T[i] - alpha_T[j-1]) / (alpha_T[j] - alpha_T[j-1]) + alpha_alpha[j-1]
    return alpha


alpha= alpha_func(T_sample_kelvin)


#calculate final C_V
kappa= 137*10**9 # Pa
V_0= 7.092*10**-6 # m^3/mol

def C_V_calc(C_p,alpha,kappa,V_0,T):
    return C_p-(9*(alpha**2)*kappa*V_0*T)


C_V= C_V_calc(C_p, alpha, kappa, V_0, T_sample_kelvin)

for e1, e2, e3 in zip(T_sample_kelvin, alpha, C_V):
    print("T/K = ", e1, "alpha = ", e2, "C_V = ", e3)

# plot C_V against T
plt.figure()
plt.plot(T_sample_kelvin, C_V, '.k', label="Data")

plt.xlabel(r'$T\,/\,\mathrm{K}$')
plt.ylabel(r'$C_V\,/\,\mathrm{\frac{J}{K mol}}$')

plt.grid("True")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("build/CV_plot.pdf")
#plt.show()

###################### TABLE ##########################
titles=[r"$t/\unit{\second}$",                              # t
        r"$\upDelta t/\unit{\second}$",                     # dt
        r"$U/\unit{\volt}$",                                # U
        r"$I/\unit{\milli\volt}$",                          # I
        r"$H/\unit{\joule}$",                               # H
        r"$R\idx{sample}/\unit{\ohm}$",                     #R_sample
        r"$T\idx{sample}/\unit{\kelvin}$",                  #T_sample
        r"$\upDelta T\idx{sample}/\unit{\kelvin}$",         #dT_sample#
        r"$C\idx{p}/\unit{\frac{\joule}{\mole\kelvin}}$",    #C_p
        r"$R\idx{shield}/\unit{\ohm}$",                     #R_shield
        r"$T\idx{shield}/\unit{\kelvin}$",                  #T_shield
        r"$\alpha/ 10^{-6}\unit{\per\kelvin}$",             #alpha
        r"$C\idx{V}/\unit{\frac{\joule}{\mole\kelvin}}$"]    #C_V



t=np.round(t, decimals=2)
dt=np.round(dt, decimals=2)
I=I*10**3
I=np.round(I, decimals=1)
H=np.round(H, decimals=2)
T_sample_kelvin=np.round(T_sample_kelvin, decimals=2)
dT_sample_kelvin=np.round(dT_sample_kelvin, decimals=2)
C_p = np.round(C_p, decimals=2)
T_shield_kelvin= np.round(T_shield_kelvin, decimals=2)
alpha= alpha*10**6
alpha=np.round(alpha, decimals=2)
C_V = np.round(C_V, decimals=2)

new_data = np.column_stack((t,dt,U,I,H,R_sample,T_sample_kelvin,dT_sample_kelvin,C_p,R_shield,T_shield_kelvin,alpha,C_V))
print(C_V)
generatetable(new_data, "all",
              r"All measured and calculated datas.","table_all",
              titles, True) 

#################### Debye Temperature #######################
print(C_V[0:9])
print(T_sample_kelvin[0:9])
#[  nan 14.51 15.78 16.78 17.91 15.7  21.26 20.2  17.98]
thetaD_per_T= np.array([np.nan,3.5,3.2,2.9,2.8,3.2,1.8,2.1,2.7])
thetaD= thetaD_per_T*T_sample_kelvin[0:9]
thetaD = np.round(thetaD, decimals=2)
print(thetaD)
thetaD_mean= np.round(np.mean(thetaD[1:]),decimals=2)
thetaD_std= np.round(np.std(thetaD[1:]),decimals=2)
print("thetaD_mean= ", thetaD_mean, " pm ", thetaD_std)


########################## table ########################
theta_data= np.column_stack((t[0:9],T_sample_kelvin[0:9],C_V[0:9], thetaD_per_T, thetaD))
theta_titles = [r"$t/\unit{\second}$", 
                r"$T/\unit{\kelvin}$",
                r"$C\idx{V}/\unit{\frac{\joule}{\mole\kelvin}}$",
                r"$\frac{\theta\idx{D}}{T}$",
                r"$\theta\idx{D}/\unit{\kelvin}$"]

generatetable(theta_data, "thetaD",
              r"Measured and calculated datas for the Debye temperature.","thetaD_table",
              theta_titles, False) 

##################### Theory ###########################
vL= 4.7*10**3 # m/s
vT= 2.26*10**3 # m/s
NA= 6.022 *10**23 # 1/mol

def omegaD_calc(NA,V0,vL,vT):
    return (((18*(np.pi**2)*NA)/V0) * ((1/vL**3)+ (2/vT**3))**-1)**(1/3)

omegaD= omegaD_calc(NA,V_0,vL,vT)
print(np.round(omegaD*10**-13, decimals=2))
#5,391317262824967*10**13

x_th =  332.18
x_exp = 347.11

dev = ((x_exp-x_th)/x_th)*100 #%
print("the deviation is: ", dev)