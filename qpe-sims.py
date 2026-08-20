import pennylane as qml
from pennylane import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from pennylane.labs import resource_estimation as plre
import tikzplotlib
##############

def H_ph_MAKER(nstates, w0):
    h1=qml.BoseWord({(0, 0): "+", (1, 0): "-"})
    h2=qml.BoseWord({(0, 0): "-", (1, 0): "+"})
    
    h=qml.BoseSentence({h1:w0, h2:w0})
    return qml.unary_mapping(h, nstates)

def H_e_MAKER(sites, t):
    # fermi_words=[-t*qml.FermiC(k)*qml.FermiA(k+1) for k in range(sites-1)]
    # print(fermi_words)
    # fermi_word=sum(fermi_words)-t*qml.FermiC(sites-1)*qml.FermiA(0)
    # return qml.jordan_wigner(fermi_word)
    # # for k in range(sites-1):
    # #     print(k, k+1)
    # #     fermi_word+=-t*qml.FermiC(k)*qml.FermiA(k+1)
    # print(fermi_word)
    # print(-t*qml.FermiC(0)*qml.FermiA(1))
    return qml.jordan_wigner(-t*qml.FermiC(0)*qml.FermiA(1)-t*qml.FermiC(1)*qml.FermiA(0))


def H_Maker(esites, m, g, w0=1, t=1):
    ###esites number of electronic sites (each 1qubit)
    ###m truncated hilbert space for one phonon mode
    ###g, w0, t: parameters in H

    ##first qubits store the electronic dof
    fullham=H_e_MAKER(esites, t) #0.5 * qml.PauliX(0) @ qml.PauliX(1)
    # gottverlassendict={}
    # for k in range(esites):
    #     gottverlassendict.update({k:esites})
    ##now we enciode the bosonic phonon modes
    for j in range(esites):
        babyindexham=H_ph_MAKER(m, w0)
        godforsakendict={}
        for k in range(m):
            godforsakendict.update({k:esites+m*j+k})
        
        fullham+=qml.map_wires(babyindexham, godforsakendict)
        ###now do the interactions###
        nqubit=qml.jordan_wigner(g*qml.FermiC(j)*qml.FermiA(j))
        boscplng=qml.BoseSentence({qml.BoseWord({(0, 0): "+"}):1, qml.BoseWord({(0, 0): "-"}):1})
        bqubit=qml.map_wires(qml.unary_mapping(boscplng, m), godforsakendict)
        fullham+=nqubit@bqubit
        
    
    return fullham
        
# fullH=qml.simplify(H_Maker(2, 2, 1,1/2, 1))
fullH=qml.simplify(H_Maker(2, 6,g=10, w0=1,t=1.5))
# print(fullH)
# print(qml.matrix(fullH)-np.conj(qml.matrix(fullH)).T)
trottprod=qml.TrotterProduct(fullH, time=1, order=2)

from pennylane.templates import QuantumPhaseEstimation
import pennylane.estimator as qre

# Tcount=np.array([1.518E+4, 2.257E+4, 2.996E+4]) #for m=2
Tcount=[]
n_est=5 #around 0.03 precision
m=2
b=2
fullH=qml.simplify(H_Maker(2, 2,1, 1,0))
trottprod=qml.TrotterProduct(fullH, time=1, order=2)
print(trottprod)
dev = qml.device("default.qubit", wires=range(n_est+b+m*b+2))
@qml.qnode(dev)
def circuit():
     
    QuantumPhaseEstimation(trottprod,
        estimation_wires=[6, 7, 9, 10, 12],
    )
    
    return qml.probs(range(b+m*b, b+m*b+n_est))


with qml.Tracker(dev) as tracker:
    circuit()


res = qre.estimate(circuit, gate_set={"Hadamard", "S", "T", "CNOT", "Toffoli"})()
print(res)

# plt.plot([4, 8, 16], np.log(Tcount))


# resources_lst = tracker.history['resources']
# print(resources_lst[0])