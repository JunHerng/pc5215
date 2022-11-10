"""Code to evolve a wavepacket.
Written for NUS PC5215.
Author: Gan Jun Herng.
Date: 10/11/2022.
"""
import numpy as np
import cmath # Complex number operations
import matplotlib.pyplot as plt

################################
# Generate gaussian wavepacket #
################################

# Physical Variables
hbar=1
m = 2
a = 5 # potential barrier width
v_base = 1
v0 = 1*v_base # barrier height
# Gaussian parameters
sigma=a*2
x0 = -(4*sigma)
p0 = np.sqrt(2*m*v_base/2)
e = p0**2/(2*m)

num_bins = 1000
bins = np.linspace(x0-4*sigma,a+6*sigma,num_bins)
delta_x = bins[1]-bins[0]
delta_t = 0.1

def psi(x,p=p0):
    return np.exp(-((x-x0)**2)/(2*sigma**2) + 1j*p*(x-x0)/hbar)

#####################
# Evolve wavepacket #
#####################
def gen_t_matrix(wavepkt):
    "Generates central difference matrix"
    size = len(wavepkt)
    t_matrix = np.zeros((size,size))
    for i in range(size):
        if i==0: #First row
            t_matrix[i][0] = 2
            t_matrix[i][1] = -1
        elif i==size-1: #Last row
            t_matrix[i][-1] = 2
            t_matrix[i][-2] = -1
        else:
            t_matrix[i][i] = 2
            t_matrix[i][i-1] = -1
            t_matrix[i][i+1] = -1
    t = hbar**2/(2*m*delta_x**2)
    print(f't: {t}')
    return t_matrix*t

def find_nearest(array,value):
    "Helper function for gen_v_matrix"
    array=np.array(array)
    idx= np.argmin(np.abs(array-value))
    return idx

def gen_v_matrix(wavepkt, type='constant'):
    "Generates diagonal V matrix"
    size = len(wavepkt)
    v_matrix = np.zeros((size,size))
    # Determine start/stop idx of barrier
    pos_array = bins
    barrier_width = a
    start_idx = find_nearest(pos_array,0)
    end_idx = find_nearest(pos_array,barrier_width)
    # Keep potential strictly positive 
    # for quadratic case
    while pos_array[start_idx] < 0:
        start_idx +=1
    while pos_array[end_idx] > a:
        end_idx -=1
    for i in range(size):
        if i >= start_idx and i <= end_idx:
            if type=='constant':
                v_matrix[i][i] = v0
            elif type=='quadratic':
                v_matrix[i][i] = 4*v0*(a-bins[i])*bins[i]/a**2
    return v_matrix

def crank_nicholson(wavepkt, type='constant', delta_t=delta_t,n=1):
    "Evolves wavepacket by n timesteps"
    start_sum = np.array([a*a.conjugate() for a in wavepkt]).sum()
    h = gen_t_matrix(wavepkt) + gen_v_matrix(wavepkt,type=type)
    coeff = delta_t/(2*hbar)*1j
    iden = np.identity(len(wavepkt))
    m1 = iden - coeff*h
    m2 = iden + coeff*h
    m2_inv = np.linalg.inv(m2)
    u = np.matmul(m2_inv,m1)
    for i in range(n):
        wavepkt = u.dot(wavepkt)
    end_sum = np.array([a*a.conjugate() for a in wavepkt]).sum()
    # Check normalization preserved
    print(f'norm check - start sum: {start_sum}, end sum: {end_sum}')
    return wavepkt

def get_transmission(energy):
    # Energy to p
    p = np.sqrt(2*m*energy)

    # Evolve wavepakcet to a+sigma
    dist = a+5*sigma-x0
    n = int(dist/(p/m*delta_t))
    print(f'n:{n}')
    wavepkt = [psi(x,p) for x in bins]
    norm_factor = np.sqrt(np.array([a*a.conjugate() for a in wavepkt]).sum())
    wavepkt = np.array(wavepkt)/norm_factor
    wavepkt = crank_nicholson(wavepkt,type='quadratic',n=n)

    # Locate idx of a
    idx = find_nearest(bins, a)

    # Calculate T
    t_packet = wavepkt[idx:]
    T = np.array([a*a.conjugate() for a in t_packet]).sum()
    T = T.real
    print(f'energy: {energy}  T:{T}')
    return T, wavepkt

###################
# Plot wavepacket #
###################
def plot_psisquared(wavepkt):
    psi_sq=[a*a.conjugate() for a in wavepkt]
    plt.plot(bins,psi_sq)
    plt.show()

def plot_real(wavepkt):
    psi_real=[a.real for a in wavepkt]
    plt.plot(bins,psi_real)
    plt.show()


# Plot wavepacket T(E)
# e_arr = np.linspace(0.01,2,100)
# t_arr = np.zeros_like(e_arr)
# for i in range(len(e_arr)):
#     trans, pkt = get_transmission(e_arr[i])
#     t_arr[i] = trans

# print(t_arr)
# with open('tmp.dat','a+') as f:
#     for data in t_arr:
#         f.write(str(data))

# get data
# with open('500_data.txt','r') as f:
#     a = f.readlines()

# a = [ln.strip('[') for ln in a]
# a = [ln.strip(']') for ln in a]
# a = [ln.strip().split() for ln in a]
# b = []
# for i in a:
#     for j in i:
#         b.append(j)
# b = [float(i) for i in b]

#plt.plot(e_arr,t_arr,label="Wave packet")

# Plot plane wave T(E)
def sinh_func(x):
    ans = 1/(1+ ( (np.sinh(10*(1-x))**2) / (4*(1-x)*x) ) )
    return ans

def sin_func(x):
    ans = 1/(1+ ( (np.sin(10*(x-1))**2) / (4*(x-1)*x) ) )
    return ans

# x = np.linspace(0.01,2,100)
# y = np.zeros_like(x)

# for i in range(50):
#     y[i] = sinh_func(x[i])

# for i in range(50,100):
#     y[i] = sin_func(x[i])

# plt.plot(e_arr,y,label='Plane wave')
# plt.xlabel(r'$E/V_0$')
# plt.ylabel('T')
# plt.legend()
# plt.title(r'Graph of T vs $E/V_0$')
# plt.show()

# Time evolution multiplot
pp = np.sqrt(2*m/2)
v0 = pp/m
e = pp**2/(2*m)
dist = a+sigma-x0
n_traverse = int(dist/(v0*delta_t))
fig, axs = plt.subplots(2,2)
n1,n2,n3 = 300,400,500
xlim, ylim = 0,1

print(f'Energy: {e}, V0: {v0}, a: {a}, v0: {pp/m}, sigma: {sigma}')
print(f'start:{x0}, dist:{dist}, end_ideal:{x0+dist}, n_tra: {n_traverse}, end_actual:{x0+n_traverse*delta_t*pp/m}')

wavepkt = [psi(x,pp) for x in bins]
norm_factor = np.sqrt(np.array([a*a.conjugate() for a in wavepkt]).sum())
wavepkt = np.array(wavepkt)/norm_factor
ylim = wavepkt.max()*1.5

y0 = [a*a.conjugate() for a in wavepkt]
y0b = [a.real for a in wavepkt]
axs[0,0].plot(bins,y0b)
axs[0,0].set_ylim([0,ylim])
axs[0,0].vlines(0,0,ylim,linestyles="dotted",alpha=0.5,colors='k')
axs[0,0].vlines(5,0,ylim,linestyles="dotted",alpha=0.5,colors='k')
axs[0,0].set_title(f'n=0')

wavepkt = crank_nicholson(wavepkt,n=n1)
y1 = [a*a.conjugate() for a in wavepkt]
y1b = [a.real for a in wavepkt]
axs[0,1].plot(bins,y1b)
axs[0,1].set_ylim([0,ylim])
axs[0,1].vlines(0,0,ylim,linestyles="dotted",alpha=0.5,colors='k')
axs[0,1].vlines(5,0,ylim,linestyles="dotted",alpha=0.5,colors='k')
axs[0,1].set_title(f'n={n1}')

wavepkt = crank_nicholson(wavepkt,n=n2)
y2 = [a*a.conjugate() for a in wavepkt]
y2b = [a.real for a in wavepkt]
axs[1,0].plot(bins,y2b)
axs[1,0].set_ylim([0,ylim])
axs[1,0].vlines(0,0,ylim,linestyles="dotted",alpha=0.5,colors='k')
axs[1,0].vlines(5,0,ylim,linestyles="dotted",alpha=0.5,colors='k')
axs[1,0].set_title(f'n={n2}')

wavepkt = crank_nicholson(wavepkt,n=n3)
y3 = [a*a.conjugate() for a in wavepkt]
y3b = [a.real for a in wavepkt]
axs[1,1].plot(bins,y3b)
axs[1,1].set_ylim([0,ylim])
axs[1,1].vlines(0,0,ylim,linestyles="dotted",alpha=0.5,colors='k')
axs[1,1].vlines(5,0,ylim,linestyles="dotted",alpha=0.5,colors='k')
axs[1,1].set_title(f'n={n3}')


plt.suptitle(fr'Plot of Re($\Psi$) after n time steps, $\Delta t$ = {delta_t}')
plt.tight_layout()
plt.show()
