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
Tcountomega=[]
# eregisterqubitsrange=[ 6, 7, 8, 9, 10]
eregisterqubitsrange=[2, 3, 5]

def binary(num, length=4):
    """BASE 10 TO BASE 2 FCN, ADMITTEDLY FROM STACK EXCHANGE"""
    return format(num, '#0{}b'.format(length + 2)).replace('0b', '')


# ####QSP BASED ESTIMATE####
for ind, eregisterqubits in enumerate(eregisterqubitsrange):
    qspres=plre.estimate(QAA_QSP_NONHERMITIAN_LCU_CIRCUIT, my_gate_set)(4, 22, truncatedspacequbits[0])
    qffres=plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(0,  ctrlqubits=eregisterqubits, ctrlzeros=eregisterqubits)
    
    for j in range(1, 2**eregisterqubits):
        zerocount=binary(j, eregisterqubits).count('0')
        qffres+=plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(truncatedspacequbits[0],  ctrlqubits=eregisterqubits, ctrlzeros=zerocount)
    totalcount=(2**eregisterqubits)*qspres+qffres
    # Tcountsqsp.append(qffres.clean_gate_counts["T"])
    Tcountsqsp.append(totalcount.clean_gate_counts["T"])
    Tcountomega.append((2**eregisterqubits)*qspres.clean_gate_counts["T"])


# plt.plot(np.array([64, 2**7, 2**8, 2**9, 2**10]), np.log10(Tcountsqsp)-np.log(Tcountomega), marker='x', label=r'$U_{\alpha}$')
# plt.plot(np.array([2**6, 2**7, 2**8, 2**9, 2**10]), np.log10(Tcountomega), marker='x',label=r"$\Omega$")
# plt.plot(np.array([2**6, 2**7, 2**8, 2**9, 2**10]), np.log10(Tcountsqsp), marker='x',label=r"$\Omega$")

# plt.xlabel("Log of number of sites")
# plt.ylabel("T Count")
pathname="pennylanecircuits.py"
current_path=os.path.abspath(__file__)
coeff_path=current_path.replace(pathname, "")
save_path = os.path.normpath(coeff_path)

# tikzplotlib.save(os.path.join(save_path, "comp_langfirsov_vacumm.tex"), flavor="context")

# plt.legend()
# plt.show()

# alphalist=[]
# N2=[]
# N4=[]
# N8=[]
# N16=[]
# N32=[]

# alphalistvac=[]
# N2vac=[]
# N4vac=[]
# N8vac=[]
# N16vac=[]
# N32vac=[]
# with open('C:/Users/skelt/Documents/Github/warm-start/Overlap_Omega=1_alpha_modified.txt') as infile:
#     for line in infile:
#         alphalist.append(line.split()[0])
#         N4.append(float(line.split()[2]))
#         N8.append(float(line.split()[3]))
#         N16.append(float(line.split()[4]))
#         N2.append(float(line.split()[1]))

# with open('C:/Users/skelt/Documents/Github/warm-start/Overlap_NonInteracting_L_Om1_nph20.txt') as infile:
#     for line in infile:
#         alphalistvac.append(line.split()[0])
#         N4vac.append(float(line.split()[2]))
#         N8vac.append(float(line.split()[3]))
#         N16vac.append(float(line.split()[4]))
#         N2vac.append(float(line.split()[1]))

# # print(np.array(alphalist)-np.array(alphalistvac))

# N2comp=((Tcountomega[0]/Tcountsqsp[0])**2)*np.divide(np.array(N2), np.array(N2vac))
# N4comp=((Tcountomega[1]/Tcountsqsp[1])**2)*np.divide(np.array(N4), np.array(N4vac))
# N8comp=((Tcountomega[2]/Tcountsqsp[2])**2)*np.divide(np.array(N8), np.array(N8vac))
# N16comp=((Tcountomega[3]/Tcountsqsp[3])**2)*np.divide(np.array(N16), np.array(N16vac))

lambdalist=[]
N405a0=[]
N405a10=[]
N805a0=[]
N805a10=[]

N410a0=[]
N410a10=[]
N810a0=[]
N810a10=[]

with open('C:/Users/skelt/Documents/Github/warm-start/Alberto_probs_N8_W1') as infile:
    for line in infile:
        lambdalist.append(line.split()[0])
        N810a0.append(float(line.split()[3]))
        N810a10.append(float(line.split()[2]))
with open('C:/Users/skelt/Documents/Github/warm-start/Alberto_probs_N4_W1') as infile:
    for line in infile:
        N410a0.append(float(line.split()[3]))
        N410a10.append(float(line.split()[2]))

with open('C:/Users/skelt/Documents/Github/warm-start/Alberto_probs_N8_W05') as infile:
    for line in infile:
        N805a0.append(float(line.split()[3]))
        N805a10.append(float(line.split()[2]))

with open('C:/Users/skelt/Documents/Github/warm-start/Alberto_probs_N4_W05') as infile:
    for line in infile:
        N405a0.append(float(line.split()[3]))
        N405a10.append(float(line.split()[2]))

N405comp=(Tcountomega[0]/Tcountsqsp[0])*np.divide(np.array(N405a10), np.array(N405a0))
N410comp=(Tcountomega[0]/Tcountsqsp[0])*np.divide(np.array(N410a10), np.array(N410a0))
N805comp=(Tcountomega[1]/Tcountsqsp[1])*np.divide(np.array(N805a10), np.array(N805a0))
N810comp=(Tcountomega[1]/Tcountsqsp[1])*np.divide(np.array(N810a10), np.array(N810a0))
N3205comp=(Tcountomega[2]/Tcountsqsp[2])*np.divide(np.array(N805a10), np.array(N805a0))
N3210comp=(Tcountomega[2]/Tcountsqsp[2])*np.divide(np.array(N810a10), np.array(N810a0))

plt.plot(np.array(lambdalist), (1/np.array(N405comp)), label='405')
plt.plot(np.array(lambdalist), (1/np.array(N410comp)), label='410')
plt.plot(np.array(lambdalist), (1/np.array(N805comp)), label='805')
plt.plot(np.array(lambdalist), (1/np.array(N810comp)), label='810')
plt.plot(np.array(lambdalist), (1/np.array(N3205comp)), label='3205')
plt.plot(np.array(lambdalist), (1/np.array(N3210comp)), label='3210')

tikzplotlib.save(os.path.join(save_path, "ratio_bothw_alphavacvseq10.tex"), flavor="context")

plt.legend()
plt.show()
