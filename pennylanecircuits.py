import pennylane as qml
from pennylane import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from pennylane.labs import resource_estimation as plre
import tikzplotlib
##############

def CTRL_FFTBASED_CIRCUIT(systq,  ctrlqubits=1, ctrlzeros=0):
    plre.ResourceQFT(systq, wires=range(0, systq))

    ####CONTROLLED ROTATIONS
    for j in range(systq):
        plre.ResourceControlled(plre.ResourceRY(wires=j), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq,systq+ctrlqubits))
    
    plre.ResourceQFT(systq, wires=range(0, systq))
    return

def QSP_NONHERMITIAN_LCU_CIRCUIT(d, systq, beq=1, qspq=2):
    plre.ResourceHadamard(wires=[0])

    ###dth iteration##
    plre.ResourceMultiControlledX(2, 2, wires=range(0, qspq+beq))
    plre.ResourceRZ(wires=[0])
    plre.ResourceMultiControlledX(2, 2, wires=range(0, qspq+beq))
    plre.ResourceHadamard(wires=[1])

    ####CONTROLLED BLOCK-ENCODING
    plre.ResourceControlled(plre.ResourceRY(wires=[qspq]), num_ctrl_wires=1, num_ctrl_values=1, wires=range(qspq-1, qspq+beq))
    for j in range(systq):
        plre.ResourceControlled(plre.ResourceCRY(wires=[qspq, qspq+beq+j]) , num_ctrl_wires=1, num_ctrl_values=1, wires=[qspq-1, qspq, qspq+beq+j]  )
    plre.ResourceControlled(plre.ResourceX(wires=[qspq]), num_ctrl_wires=1, num_ctrl_values=1,  wires=range(qspq-1, qspq+beq)  )
    ###CONTROLLED BLOCK-ENCODING CONJUGATE
    plre.ResourceControlled(plre.ResourceX(wires=[qspq]), num_ctrl_wires=1, num_ctrl_values=0, wires=range(qspq-1, qspq+beq) )
    for j in range(systq):
        plre.ResourceControlled(plre.ResourceCRY(wires=[qspq, qspq+beq+j]), num_ctrl_wires=1, num_ctrl_values=0,wires=[qspq-1, qspq, qspq+beq+j] )
    plre.ResourceControlled(plre.ResourceRY(wires=[qspq]), num_ctrl_wires=1, num_ctrl_values=0,wires=range(qspq-1, qspq+beq) )
    ###END OF BLOCK-ENCODINGS
    
    plre.ResourceX(wires=[1])
    #all of the next iterations
    for j in range(1, d):
        plre.ResourceZ(wires=[0])
        plre.ResourceHadamard(wires=[1])

        plre.ResourceMultiControlledX(2, 2, wires=range(0, qspq+beq))
        plre.ResourceRZ(wires=[0])
        plre.ResourceMultiControlledX(2, 2, wires=range(0, qspq+beq))
        plre.ResourceHadamard(wires=[1])

        ####CONTROLLED BLOCK-ENCODING
        plre.ResourceControlled(plre.ResourceRY(wires=[qspq]), num_ctrl_wires=1, num_ctrl_values=1, wires=range(qspq-1, qspq+beq))
        for j in range(systq):
            plre.ResourceControlled(plre.ResourceCRY(wires=[qspq, qspq+beq+j]) , num_ctrl_wires=1, num_ctrl_values=1, wires=[qspq-1, qspq, qspq+beq+j]  )
        plre.ResourceControlled(plre.ResourceX(wires=[qspq]), num_ctrl_wires=1, num_ctrl_values=1,  wires=range(qspq-1, qspq+beq)  )
        ###CONTROLLED BLOCK-ENCODING CONJUGATE
        plre.ResourceControlled(plre.ResourceX(wires=[qspq]), num_ctrl_wires=1, num_ctrl_values=0, wires=range(qspq-1, qspq+beq) )
        for j in range(systq):
            plre.ResourceControlled(plre.ResourceCRY(wires=[qspq, qspq+beq+j]), num_ctrl_wires=1, num_ctrl_values=0,wires=[qspq-1, qspq, qspq+beq+j] )
        plre.ResourceControlled(plre.ResourceRY(wires=[qspq]), num_ctrl_wires=1, num_ctrl_values=0,wires=range(qspq-1, qspq+beq) )
        ###END OF BLOCK-ENCODINGS
        plre.ResourceX(wires=[1])

    ###0th ITERATION##
    plre.ResourceZ(wires=[0])
    plre.ResourceHadamard(wires=[1])

    plre.ResourceMultiControlledX(2, 2,wires=range(0, qspq+beq))
    plre.ResourceRZ(wires=[0])
    plre.ResourceMultiControlledX(2, 2,wires=range(0, qspq+beq))
    plre.ResourceHadamard(wires=[1])

    ##LAST LCU STEP
    plre.ResourceHadamard(wires=[0])
    return

def QAA_QSP_NONHERMITIAN_LCU_CIRCUIT(m, d, systq, beq=1, qspq=2, qaaq=1, ):
    for k in range(systq):
        plre.ResourceHadamard(wires=[qspq+beq+k])
    plre.ResourceRY(wires=[qspq+beq+systq])
    QSP_NONHERMITIAN_LCU_CIRCUIT(d, systq, beq=1, qspq=2)

    ####ITERATION STARTS HERE###
    for l in range(m):
        plre.ResourceControlled(plre.ResourceRZ(wires=[qspq+beq+systq]), num_ctrl_wires=qspq+beq, num_ctrl_values=qspq+beq, wires=range(0, qspq+beq)) #rotation around desired projector
        plre.ResourceRY(wires=[qspq+beq+systq])
        QSP_NONHERMITIAN_LCU_CIRCUIT(d, systq, beq=1, qspq=2) ##dagger will cost the same
        for k in range(systq):
            plre.ResourceHadamard(wires=[qspq+beq+k]) 
        plre.ResourceControlled(plre.ResourceRZ(wires=[qspq+beq+systq]), num_ctrl_wires=qspq+beq+systq, num_ctrl_values=qspq+beq+systq, wires=range(0, qspq+beq+systq)) #over all qubits

        for j in range(systq):
            plre.ResourceHadamard(wires=[qspq+beq+j])
        plre.ResourceRY(wires=[qspq+beq+systq])
        QSP_NONHERMITIAN_LCU_CIRCUIT(d, systq, beq=1, qspq=2) 
    
    return


my_gate_set = {"Hadamard", "S", "T", "CNOT", "Toffoli"}
# my_gate_set = {"Hadamard", "S", "T", "CNOT"}


truncatedspacequbits=np.array([6])
Tcountsqsp=[]
eregisterqubitsrange=[5, 6, 7, 8, 9, 10]

def binary(num, length=4):
    """BASE 10 TO BASE 2 FCN, ADMITTEDLY FROM STACK EXCHANGE"""
    return format(num, '#0{}b'.format(length + 2)).replace('0b', '')


####QSP BASED ESTIMATE####
for ind, eregisterqubits in enumerate(eregisterqubitsrange):
    qspres=plre.estimate(QAA_QSP_NONHERMITIAN_LCU_CIRCUIT, my_gate_set)(4, 22, truncatedspacequbits[0])
    qffres=plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(0,  ctrlqubits=eregisterqubits, ctrlzeros=eregisterqubits)
    
    for j in range(1, 2**eregisterqubits):
        zerocount=binary(j, eregisterqubits).count('0')
        qffres+=plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(truncatedspacequbits[0],  ctrlqubits=eregisterqubits, ctrlzeros=zerocount)
    totalcount=(2**eregisterqubits)*qspres+qffres
    Tcountsqsp.append(totalcount.clean_gate_counts["T"])


plt.plot(eregisterqubitsrange, (Tcountsqsp), marker='x', label='QET')

plt.xlabel("Log of number of sites")
plt.ylabel("T Count")
pathname="pennylanecircuits.py"
current_path=os.path.abspath(__file__)
coeff_path=current_path.replace(pathname, "")
save_path = os.path.normpath(coeff_path)
tikzplotlib.save(os.path.join(save_path, "qsp_22_scale.tex"), flavor="context")

# tikzplotlib.save(os.path.join(save_path, "comp_qsp_vqa_22_logscale.tex"), flavor="context")

plt.legend()
plt.show()