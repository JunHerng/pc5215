"""
Written for PC5215 Lab 2 to compute the energy of a H_2 molecule in ground state.
"""

from concurrent.futures import process
from hashlib import new
import math
import random
from turtle import st
import numpy as np
import matplotlib.pyplot as plt

def move_electron(e: list, bound: float = 12):
    """Given current electron position (input list),
    return updated position.

    Args:
        e (list): Electron's current position in 3D.
        bound (float): Set search bound. 
    Returns:
        a (list): Electron's updated position.
    """
    a = e.copy()
    for i in range(len(a)):
        coord = a[i]
        #new_coord = random.uniform(coord-1,coord+1) # Search mthd 1
        new_coord = coord + (2*random.random()-1)*0.5 # Search mthd 2
        a[i] = new_coord
    return a

def compute_psi_squared(x1, x2, r_ab):
    e1 = np.array(x1)
    e2 = np.array(x2)
    r_b = np.array([r_ab,0,0])

    # Compute relative distances
    r_a1 = np.linalg.norm(e1)
    r_a2 = np.linalg.norm(e2)
    b1 = r_b-e1
    r_b1 = np.linalg.norm(b1)
    b2 = r_b-e2
    r_b2 = np.linalg.norm(b2)

    # Compute psi_squared
    result = math.exp(2*(-r_a1-r_b2)) + math.exp(2*(-r_a2-r_b1)) \
             + 2*math.exp(-r_a1 - r_a2 - r_b1 - r_b2)
    
    return result

def compute_acceptance(psi_prime, psi_n):
    ratio = psi_prime/psi_n
    r = min(1,ratio)
    return r

def process_acceptance(r):
    xi = random.random()
    if r > xi: # Update value
        return True
    else: # Reject value
        return False

def compute_fx(x1,x2,r_ab):
    e1 = np.array(x1)
    e2 = np.array(x2)
    r_b = np.array([r_ab,0,0])

    # Compute relative distances
    r_a1 = np.linalg.norm(e1)
    r_a2 = np.linalg.norm(e2)
    b1 = r_b-e1
    r_b1 = np.linalg.norm(b1)
    b2 = r_b-e2
    r_b2 = np.linalg.norm(b2)
    vec_12 = e1-e2
    r_12 = np.linalg.norm(vec_12)

    # Set up fx components
    num1 = math.exp(-r_a1-r_b2); num2 = 1/r_a1 + 1/r_b2 - 1
    num3 = math.exp(-r_b1-r_a2); num4 = 1/r_b1 + 1/r_a2 - 1
    denom = num1 + num3
    v = -1/r_a1-1/r_a2-1/r_b1-1/r_b2+1/r_ab+1/r_12
    # Bring it all in
    result = (num1*num2 + num3*num4)/denom + v
    #print(f'fx: {result}')
    return result

def jackknife(fx_array):
    """Takes in array of fx values, splits it into 100 segments.
    Calculates expectation and std.dev based on jacknife method.

    Args:
        fx_array (np.array): array of fx values
    """
    length = len(fx_array)
    seg_length = length/100
    arr = np.array([])
    u_0 = np.average(fx_array)
    for i in range(100):
        start_idx = int((i)*seg_length)
        end_idx = start_idx + 100
        if i == 0:
            knifed_array = fx_array[end_idx:]
        elif i == 99:
            knifed_array = fx_array[:start_idx]
        else:
            knifed_array = np.concatenate((fx_array[:start_idx],fx_array[end_idx:]))
        avg = np.average(knifed_array)
        arr = np.append(arr,avg)
    u_bar = np.average(arr)
    expected_u = u_0 - 99*(u_bar-u_0)
    tmp = arr - u_bar
    delta_u = math.sqrt((99/100)*np.sum(tmp**2))

    return expected_u,delta_u

def metropolis(r_ab: float=1.0, iter: int = 1000):
    """Uses the Metropolis algorithm to find the expectation
    value of a hydrogen molecule's energy.

    Args:
        r_ab (float): Sets the proton separation in units of Bohr radius.
                      Default 1.0. Which is about 0.529 angstrom.
        iter (int):   Number of iterations to run before terminating.
                      Default 1000. Min 100 due to implementation of
                      jackknife().
    """
    # Initial Positions
    # Proton
    ra = [0,0,0]
    rb = [r_ab,0,0] # r_ab is the distance between the 2 protons
    # Electron
    e1 = [1,1,1]
    e2 = [-1,-1,-1]

    fx_list = []
    i = 0
    # Metropolis algo
    while i < iter:
        # Current position
        e1_old = e1.copy()
        e2_old = e2.copy()
        fx_old = compute_fx(e1_old,e2_old,r_ab)
        psi_old = compute_psi_squared(e1_old,e2_old,r_ab)
        # Move to new position
        e1 = move_electron(e1)
        e2 = move_electron(e2)
        psi_prime = compute_psi_squared(e1,e2,r_ab)
        # Acceptance condition
        r = compute_acceptance(psi_prime,psi_old)
        a_result = process_acceptance(r)
        if a_result == False:
            # Revert to old positions and
            # add duplicate value to list
            fx_list.append(fx_old)
            e1 = e1_old.copy()
            e2 = e2_old.copy()
            continue
        elif a_result == True:
            # Keep new value for next loop
            psi_old = psi_prime
            fx = compute_fx(e1,e2,r_ab)
            fx_list.append(fx)
            fx_old=fx
        i+=1
    # Return average value of fx
    if len(fx_list) == 0:
        print('Something went wrong. No fx to process.')
    elif len(fx_list) > 0:
        fx_array = np.array(fx_list)
        # Burn in - Discard first 1/10 of array
        discard_idx = int(iter/10) 
        fx_array = fx_array[discard_idx:]
        result,err = jackknife(fx_array)
    if result:
        print(f'Metro Algo returned sum: {result}, r_ab: {r_ab}, {iter} iterations\n')
    return result, err

def metropolis2(r_ab: float=1.0, iter: int = 1000):
    """Similar to metropolis(), but does not use the jackknife
    method to find the error bars. This task is done by
    vary_distance2() instead.
    """
    # Initial Positions
    # Proton
    ra = [0,0,0]
    rb = [r_ab,0,0] # r_ab is the distance between the 2 protons
    # Electron
    e1 = [0.5,0.5,0.5]
    e2 = [-0.5,-0.5,-0.5]
    e1 = move_electron(e1)
    e2 = move_electron(e2)

    fx_list = []
    i = 0
    # Metropolis algo
    while i < iter:
        # Current position
        e1_old = e1.copy()
        e2_old = e2.copy()
        fx_old = compute_fx(e1_old,e2_old,r_ab)
        psi_old = compute_psi_squared(e1_old,e2_old,r_ab)
        # Pick new candidate
        e1 = move_electron(e1)
        e2 = move_electron(e2)
        psi_prime = compute_psi_squared(e1,e2,r_ab)
        r = compute_acceptance(psi_prime,psi_old)
        a_result = process_acceptance(r)
        if a_result == False:
            # Discard candidate
            fx_list.append(fx_old)
            e1 = e1_old.copy()
            e2 = e2_old.copy()
            continue
        elif a_result == True:
            # Keep candidate
            psi_old = psi_prime
            fx = compute_fx(e1,e2,r_ab)
            fx_list.append(fx)
            fx_old=fx
        i+=1
    # Return average value of fx
    if len(fx_list) == 0:
        print('Something went wrong. No fx to process.')
    elif len(fx_list) > 0:
        fx_array = np.array(fx_list)
        # Burn in - Discard first 1/10 of array
        discard_idx = int(iter/10) 
        fx_array = fx_array[discard_idx:]
        result = np.average(fx_array)
    if result:
        print(f'Metro Algo returned value: {result}, r_ab: {r_ab}, {iter} iterations\n')
    return result

def vary_distance(a: float,b: float, n: int=30, iter: int = 1000, filename: str = None):
    """Linearly varies inter-proton distance between a
       and b, and finds the energy at these r_ab. r_ab is 
       in units of Bohr radii. 

    Args:
        a (float): min proton distance
        b (float): max proton distance
        n (int): number of points. Defaults to 50
        iter (int): number of iterations to pass to metropolis()
        filename (str): saves arrays to file if filename is provided
    """

    if a<0 or b<0:
        print('a and b must be positive!')
        return

    dist_array = np.linspace(a,b,num=n)
    energy_list = []
    err_list = []
    for dist in dist_array:
        energy, err = metropolis(dist,iter)
        energy_list.append(energy)
        err_list.append(err)
        if type(filename) == str:
            with open(filename,'a+') as f:
                f.write(f'{dist},{energy},{err}\n')
    energy_array = np.array(energy_list)
    err_array = np.array(err_list)
    return dist_array, energy_array, err_array

def vary_distance2(a: float,b: float, n: int=50, iter: int = 1000):
    """Similar to vary_distance(), but calls metropolis() multiple
    times at each r_AB and uses these values to calculate average
    and standard deviation.
    """

    if a<0 or b<0:
        print('a and b must be positive!')
        return

    dist_array = np.linspace(a,b,num=n)
    energy_list = []
    err_list = []
    for dist in dist_array:
        tmp_energy = []
        for i in range(10):
            energy = metropolis2(dist,iter)
            tmp_energy.append(energy)
        ener_array = np.array(tmp_energy)
        ener = np.average(ener_array)
        energy_list.append(ener)
        err = np.std(ener_array)
        err_list.append(err)
    energy_array = np.array(energy_list)
    err_array = np.array(err_list)
    return dist_array, energy_array, err_array

if __name__=="__main__":
    # Conversion factors (9 d.p)
    bohr_to_ang = 0.529177210
    hartree_to_ev = 27.211386245

    n = 100
    iter = 1000
    x,y,err = vary_distance(0.1, 6, n=n, iter=iter, filename=f'0-1to6_n_{n}_iter_{iter}_fixed.csv')
    #x,y,err = vary_distance2(0.1, 6, n=n, iter=iter)
    # Convert units
    x = x*bohr_to_ang
    y = y*hartree_to_ev
    err = err*hartree_to_ev

    # Plot
    plt.errorbar(x,y,yerr=err)
    plt.xlabel(r'$r_{AB} (\AA)$')
    plt.ylabel(r'$E (eV)$')
    plt.title(r'Graph of $E (eV)$ vs $r_{AB} (\AA)$, iter=1000')
    plt.grid()
    plt.show()
