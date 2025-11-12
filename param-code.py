# -*- coding: utf-8 -*-
"""
Created on 12.11.2025
@author: skelt
"""
import pennylane as qml
from pennylane import numpy as np
import time
import pickle
import os.path
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

'''FANCY PREAMBLE TO MAKE BRAKET PACKAGE WORK NICELY'''
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{braket}')

####CONSTANTS WHICH THE USER SETS FOR EACH RUN
ifsave=True
run_vqe=False
systq=3
Ns=6

###CONSTANTS WHICH SHOULD STAY CONSISTENT
d=1
ctol=1.6*10**(-3)
mit=200

###CREATE DEVICES 
dev=qml.device('default.qubit', wires=systq)

###RANDOMIZE THE INITIAL PARAMETERS
params0all=np.random.rand(3*Ns*systq)

###DEFINE ARRAYS FOR IMPORTANT DATA
GS=[]
kits=[]
Nkenergy=[]
Nkits=[]

####Parameter Subcircuits
def xsquared_circuit(systq):
    for k in range(systq):
        plre.ResourcePhaseShift(wires=[k])
        
    for j in range(int(systq*(systq-1)/2)): ##extremly lazy strategy for the controlled phase gates
        plre.ResourceControlledPhaseShift()
    return 

def psquared_circuit(systq):
    qml.QFT(wires=range(systq),  num_wires=systq)
    for j in range(systq):
        plre.ResourcePhaseShift(wires=[j])
    for j in range(int(systq*(systq-1)/2)): ##extremly lasy strategy for the controlled phase gates
        plre.ResourceControlledPhaseShift()
    qml.QFT(wires=range(systq),  num_wires=systq)
    return 

def param_circuit(params, Ns, systq):
    for k in range(Ns):
        ###call ep^2
        psquared_circuit(systq)
        xsquared_circuit(systq)
        for l in range(systq):
            plre.ResourceRZ(wires=[3*l+0])
            plre.ResourceRX(wires=[3*l+1])
            plre.ResourceRY(wires=[3*l+2])
    return 


def HEA_circuit(param, wires, d):
    ###simplified circuit from http://arxiv.org/abs/1704.05018
    ###indexing looks a bit messy its a 1d list 
    ###as a 3d array the indexing would be [d iteration, qubit number, R number]
    ###given $\theta_{j i}^q$ $j\in{1, 2, 3}$
    ###as a 1d list the sequence is [\theta_{00}^0, \theta_{10}^0, \theta_{20}^0, \theta_{01}^0...]
    ###all zeros state
    #qml.BasisState(np.zeros(len(wires)), wires=wires)
    ###apply the first set of Euler rotations, without RZ terms
    indtrack=0
    for q in range(len(wires)):
        qml.RX(param[indtrack], wires=[q])
        qml.RZ(param[indtrack+1], wires=[q])
        indtrack=indtrack+2
    
    for i in range(d):
        U_ENT(wires)
        for q in range(len(wires)):
            qml.RZ(param[indtrack], wires=[q])
            qml.RX(param[indtrack+1], wires=[q])
            qml.RZ(param[indtrack+2], wires=[q])
            indtrack=indtrack+3

    
####cost function
@qml.qnode(dev, interface="autograd")
def cost_fnAA(param, H=Hdef, H0=H0def, s=sdef):
    """
    runs the HEA and then measures operator e^H
    param: an numpy array with tensor elements
    H: the hamiltonian required
    """ 
    HEA_circuit(param, range(qubits), d)
    return qml.expval(qml.simplify(float(1-s)*H0+float(s)*H))


####VQE SOLVERS
def kandala_VQE(param0, d, Hvqe=Hdef, cost_fc=HEA_cost_fcn, systsz=qubits, max_iterations=mit, conv_tol=ctol):
    """
    Function to run standard VQE
    param0: the initial parameter list
    d: the layers for the HEA
    Hvqe: the hamiltonian
    cost_fc: the cost function, default is the HEA
    systsz: system size, the number of qubits
    max_iterations: the maximum allowed number of iterations
    conv_tol: the convergence bound
    returns a python dictionary with the number of iterations, VQA solution, the solution parameters, the solution time, and all of the intermediate energy solutions
    """
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    energy=[]
    thetas=param0

    t0r=time.perf_counter()
    bpsteps=False
    ##actually runs each optimization step and returns new parameters
    for n in tqdm(range(max_iterations)):
        thetas, prev_energy= opt.step_and_cost(cost_fc, thetas, H=Hvqe)
        energy.append(HEA_cost_fcn(thetas,Hvqe))

        conv = np.abs(energy[-1] - prev_energy)
        if conv <= conv_tol:
            break
        
    t1r=time.perf_counter()
    DATA={'its':n+1, 'gsEest':energy[-1], 'angles':thetas, 'timer': t1r-t0r,'energies':energy}
    return DATA

