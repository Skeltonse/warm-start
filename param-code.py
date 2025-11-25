# -*- coding: utf-8 -*-
"""
Created on 12.11.2025
@author: skelt
"""
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from numpy.polynomial import hermite
import time
import pickle
import os.path
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from pennylane import jordan_wigner

###TO DO###
##check Hermite convention

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
#plt.rc('text', usetex=True)
#plt.rc('text.latex', preamble=r'\usepackage{braket}')

####CONSTANTS set by the user
systq=2
Ns=6
###RANDOMIZE THE INITIAL PARAMETERS
params0=np.random.rand(3*Ns*systq)

###CREATE DEVICES 
dev=qml.device('default.qubit', wires=systq)

####Parameter Subcircuits
def xsquared_circuit():
    for k in range(systq):
        qml.PhaseShift(phi=2**(2*k)-2**(systq+k), wires=[k])    
        for j in range(k+1, systq): 
            qml.ControlledPhaseShift(phi=2**(k+j+1), wires=[k, j])
    return 

def psquared_circuit():
    qml.QFT(wires=range(systq))
    for k in range(systq):
        qml.PhaseShift(phi=2**(2*k)-2**(systq+k), wires=[k]) 
        for j in range(k+1,systq): 
            qml.ControlledPhaseShift(phi=2**(k+j+1), wires=[k, j])
    qml.adjoint(qml.QFT)(wires=range(systq))
    return 

def param_circuit(params):
    for k in range(Ns):
        ###call ep^2
        psquared_circuit()
        xsquared_circuit()
        for l in range(systq):
            qml.RX(phi=params[3*l+0], wires=[l])
            qml.RY(phi=params[3*l+1], wires=[l])
            qml.RZ(phi=params[3*l+2], wires=[l])
    
    return 

###OPTIMIZATION with H (23) in bosonic creation/anahilation operators in Pennylane, 
    #H=1/2(P^2+X^2) with rescaling see under (24) which may be missing
def HO_op():
    ##Here H=bb^{dag}/2+b^{dag}b/2=b^{dag}b+[b, b^{dag}]/2
    h1=qml.BoseWord({(0, 0): "+", (1, 0): "-"})
    h2=qml.BoseWord({(0, 0): "-", (1, 0): "+"})
    
    h=qml.BoseSentence({h1:1/2, h2:1/2})
    return qml.unary_mapping(h, n_states=systq)

###OPTIMIZATION WITH FIDELITY. Define the state used for the fidelity calculation
#compute the discretized HG state over the grid with $N_x+1$ points
#from (34), \ket{\phi_{0N}}=\ket{\Xi_0}, the zeroth-order HG function, defined by overlap with position basis
#\braket{x_i\Xi_n}=\sqrt{Deta}\phi_0(x_i)
Nx=2**systq
Delta=np.sqrt(2*np.pi/Nx)
gridpts=np.linspace(-Nx/2, Nx/2-1, Nx, endpoint=True)
def phi0(x):
    factor=np.pi**(-1/4)*np.sqrt(1)
    return np.exp(-x**2/2)*hermite.hermval(x, 1)

xigrid=np.array([np.sqrt(Delta)*phi0(x) for x in gridpts]) ###from (A10)
xistate=np.reshape(xigrid, [Nx, 1])
xinorm=np.sqrt(np.conj(xistate).T@xistate)


####cost function
@qml.qnode(dev, interface="autograd")
def cost_fcn(params):
    """
    runs the parameterized circuit and then measures operator e^H
    params: an numpy array with tensor elements
    paramdepth: the number of times this circuit is applied
    """ 
    param_circuit(params)
    return qml.expval(HO_op())


@qml.qnode(dev, interface="autograd")
def state_circuit(params):
    """
    runs the parameterized circuit and then measures operator e^H
    params: an numpy array with tensor elements
    paramdepth: the number of times this circuit is applied
    returns full state
    """ 
    param_circuit(params)
    return qml.state()
    
#the cost fucntion as the fidelity
def cost_fcn_2(params):
    outputstate=state_circuit(params)
    return 1-qml.math.fidelity_statevector(outputstate, xigrid/xinorm)

###SPSA OPTIMIZERS###
def run_optimizer(opt, cost_function, init_param, num_steps, interval, execs_per_step):
    # Copy the initial parameters to make sure they are never overwritten
    param = init_param.copy()

    # Obtain the device used in the cost function
    dev = cost_function.device

    # Initialize the memory for cost values during the optimization
    cost_history = []
    # Monitor the initial cost value
    cost_history.append(cost_function(param))
    exec_history = [0]

    print(
        f"\nRunning the {opt.__class__.__name__} optimizer for {num_steps} iterations."
    )
    for step in range(num_steps):
        # Print out the status of the optimization
        if step % interval == 0:
            print(
                f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
                f"Cost = {cost_history[step]},"
                f"fidelity ={1-cost_fcn_2(param)}"
            )

        # Perform an update step
        param = opt.step(cost_function, param)

        # Monitor the cost value
        cost_history.append(cost_function(param))
        exec_history.append((step + 1) * execs_per_step)

    print(
        f"Step {num_steps:3d}: Circuit executions: {exec_history[-1]:4d}, "
        f"Cost = {cost_history[-1]},"
        f"fidelity ={1-cost_fcn_2(param)}"
    )
    return param

def grad(L, w, ck):
    
    # number of parameters
    p = len(w)
    
    # bernoulli-like distribution
    deltak = np.random.choice([-1, 1], size=p)
    
    # simultaneous perturbations
    ck_deltak = ck * deltak

    # gradient approximation
    DELTA_L = L(w + ck_deltak) - L(w - ck_deltak)

    return (DELTA_L) / (2 * ck_deltak)

def initialize_hyperparameters(alpha, lossFunction, w0, N_iterations):
    c=0.15 ##override from pennylane hint
    # c = 1e-2   # a small number

    # A is <= 10% of the number of iterations
    A = N_iterations*0.1

    # order of magnitude of first gradients
    magnitude_g0 = np.abs(grad(lossFunction, w0, c).mean())

    # the number 2 in the front is an estimative of
    # the initial changes of the parameters,
    # different changes might need other choices
    a = 2*((A+1)**alpha)/magnitude_g0

    return a, A, c

def SPSA(LossFunction, parameters, alpha=0.602,\
    gamma=0.101, N_iterations=int(1e4)): 
    
    # model's parameters
    w = parameters

    a, A, c = initialize_hyperparameters(
      alpha, LossFunction, w, N_iterations)
    a=0.2 #override based on pennylane hint
    for k in range(1, N_iterations):
        
        # update ak and ck
        ak = a/((k+A)**(alpha))
        ck = c/(k**(gamma))

        # estimate gradient
        gk = grad(LossFunction, w, ck)
        

        # update parameters
        w -= ak*gk
        if k%50==0:
            print(k, 1-LossFunction(w), LossFunction(w))

    return LossFunction(w)

num_steps_spsa = 1000
opt = qml.SPSAOptimizer(maxiter=num_steps_spsa, c=0.15, a=0.2)
execs_per_step = 2
params_last= run_optimizer(
    opt, cost_fcn, params0, num_steps_spsa, 20, execs_per_step
)
print('fidelity with VQE')
print(1-cost_fcn_2(params_last))

print('fidelity with fidelity optimized')
print(1-SPSA(cost_fcn_2, params0, gamma=0.101) )


####CHECK A SERIES OF N'S
# Nsarray=np.array([12, 14, 16])
# lossarray=np.zeros(len(Nsarray))
# fidelityarray=np.zeros(len(Nsarray))
# for ind, Ns in enumerate(Nsarray):
#     params0=np.random.rand(3*Ns*systq)
#     loss=SPSA(cost_fcn_2, params0, gamma=0.101)
#     lossarray[ind]=loss
#     fidelityarray[ind]=1-loss

# plt.plot(Nsarray, lossarray, label='loss', marker='o')
# plt.plot(Nsarray, fidelityarray, label='fidelity', marker='.')
# plt.title("SPSA performance vs number of steps")
# plt.legend()
# plt.show()