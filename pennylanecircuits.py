import pennylane as qml
from pennylane import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from pennylane.labs import resource_estimation as plre
import tikzplotlib
##############
systq=3 #for the electronic dof
ctrlq=0 #for the bosonic dof

#by convention, the ordering will be systq: 0..systq-1, beq: systq..systq+bqe-1, qspp:systq+bqe...
# dev=qml.device("default.mixed", wires=systq+beq+qspq+ctrlq)
# devbe=qml.device("default.mixed", wires=systq+beq)

def SINBE_NONSYMM(systq):
    """
    creates a block encoding of \sum_{x}\sin(x/N)\ket{x}\bra{x}.
    figure 1.a of Mcardle et al 2022 "quantum state preparation without coherent arithmetic"
    the factor of two on the rotation may not be trustworthy, should check
    """
    for j in range(systq):
        plre.ResourceCRY(wires=(j, systq)) #I use convention (control, target) which is what multicintrolled X uses and maybe hypothetically pennylane has consistent standards
        # qml.ctrl(qml.PauliRot(2*2**(j+1-systq), 'Y', wires=systq), j)
    plre.ResourceX(wires=systq)
    return 

def SINBE_NONSYMM_HC(systq):
    """
    creates a block encoding of \sum_{x}\sin(x/N)\ket{x}\bra{x}.
    figure 1.a of Mcardle et al 2022 "quantum state preparation without coherent arithmetic"
    the factor of two on the rotation may not be trustworthy, should check
    """
    plre.ResourceX(wires=systq)
    for j in range(systq):
        plre.ResourceCRY(wires=(j, systq)) #I use convention (control, target) which is what multicintrolled X uses and maybe hypothetically pennylane has consistent standards
        # qml.ctrl(qml.PauliRot(2*2**(j+1-systq), 'Y', wires=systq), j)
    return 

def SINBE_SYMM(systq):
    """
    creates a block encoding of \sum_{x}\sin(x/N)\ket{x}\bra{x}.
    figure 1.a of Mcardle et al 2022 "quantum state preparation without coherent arithmetic"
    the factor of two on the rotation may not be trustworthy, should check
    """
    plre.ResourceRY(wires=systq)
    for j in range(systq):
        plre.ResourceCRY(wires=(j, systq)) #I use convention (control, target) which is what multicintrolled X uses and maybe hypothetically pennylane has consistent standards
        # qml.ctrl(qml.PauliRot(2*2**(j+1-systq), 'Y', wires=systq), j)
    plre.ResourceX(wires=systq)
    return 

def SINBE_SYMM_HC(systq):
    """
    creates a block encoding of \sum_{x}\sin(x/N)\ket{x}\bra{x}.
    figure 1.a of Mcardle et al 2022 "quantum state preparation without coherent arithmetic"
    the factor of two on the rotation may not be trustworthy, should check
    """
    plre.ResourceX(wires=systq)
    for j in range(systq):
        plre.ResourceCRY(wires=(j, systq)) #I use convention (control, target) which is what multicintrolled X uses and maybe hypothetically pennylane has consistent standards
        # qml.ctrl(qml.PauliRot(2*2**(j+1-systq), 'Y', wires=systq), j)
    plre.ResourceRY(wires=systq)
    return 

def SYMQSP_CIRCUIT_REALPARITY_HERMITIAN(systq, philist, x):
    """
    reproduction of figure 16 from Dong et al 2021 "efficient..."
    """
    d=len(philist)-1
    for j in range(0, d):
        qml.PauliRot(2*philist[d-j], 'Z', wires=systq+beq+qspq)
        qml.PauliRot(2*np.arccos(x), 'X', wires=systq+beq+qspq)
    
    qml.PauliRot(2*philist[0], 'Z', wires=systq+beq+qspq)

def CTRL_QSP_NONHERMITIAN_LCU_CIRCUIT(d, systq, beq=1, qspq=2, ctrlqubits=1, ctrlzeros=0):
    plre.ResourceControlled(plre.ResourceHadamard(wires=[0]),num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )

    ###dth iteration##
    plre.ResourceMultiControlledX(2+ctrlqubits, 2+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(0, qspq+beq)))
    plre.ResourceControlled(plre.ResourceRZ(wires=[0]),num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )
    plre.ResourceMultiControlledX(2+ctrlqubits, 2+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(0,  qspq+beq)))
    plre.ResourceControlled(plre.ResourceHadamard(wires=[1]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )

    ####CONTROLLED BLOCK-ENCODING
    plre.ResourceControlled(plre.ResourceRY(wires=[qspq]), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=1+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq)))
    for j in range(systq):
        plre.ResourceControlled(plre.ResourceCRY(wires=[qspq, qspq+beq+j]) , num_ctrl_wires=1+ctrlqubits, num_ctrl_values=1+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq))  )
    plre.ResourceControlled(plre.ResourceX(wires=[qspq]), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=1+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq))  )
    ####CONTROLLED BLOCK-ENCODING CONJUGATE
    plre.ResourceControlled(plre.ResourceX(wires=qspq), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq)) )
    for j in range(systq):
        plre.ResourceControlled(plre.ResourceCRY(wires=[qspq, qspq+beq+j]), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq)) )
    plre.ResourceControlled(plre.ResourceRY(wires=[qspq]), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq)) )
    ####END OF BLOCK-ENCODINGS
    
    plre.ResourceControlled(plre.ResourceX(wires=[1]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )
    ##all of the next iterations
    for j in range(1, d):
        plre.ResourceControlled(plre.ResourceZ(wires=[0]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )
        plre.ResourceControlled(plre.ResourceHadamard(wires=[1]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )

        plre.ResourceMultiControlledX(2+ctrlqubits, 2+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(0, qspq+beq)))
        plre.ResourceControlled(plre.ResourceRZ(wires=[0]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )
        plre.ResourceMultiControlledX(2+ctrlqubits, 2+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(0, qspq+beq)))
        plre.ResourceControlled(plre.ResourceHadamard(wires=[1]) , num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )

        ####CONTROLLED BLOCK-ENCODING
        plre.ResourceControlled(plre.ResourceRY(wires=[qspq]), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=1+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq)) )
        for j in range(systq):
            plre.ResourceControlled(plre.ResourceCRY(wires=[qspq, qspq+beq+j]) , num_ctrl_wires=1+ctrlqubits, num_ctrl_values=1+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq))  )
        plre.ResourceControlled(plre.ResourceX(wires=qspq), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=1+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq))  )
        ####CONTROLLED BLOCK-ENCODING CONJUGATE
        plre.ResourceControlled(plre.ResourceX(wires=qspq), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq)) )
        for j in range(systq):
            plre.ResourceControlled(plre.ResourceCRY(wires=[qspq, qspq+beq+j]), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq)) )
        plre.ResourceControlled(plre.ResourceRY(wires=qspq), num_ctrl_wires=1+ctrlqubits, num_ctrl_values=ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(1, 1+beq)) )
        ####END OF BLOCK-ENCODINGS
        plre.ResourceControlled(plre.ResourceX(wires=[1]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )

    ###0th ITERATION##
    plre.ResourceControlled(plre.ResourceZ(wires=[0]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )
    plre.ResourceControlled(plre.ResourceHadamard(wires=[1]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )

    plre.ResourceMultiControlledX(2+ctrlqubits, 2+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(0, qspq+beq)))
    plre.ResourceControlled(plre.ResourceRZ(wires=[0]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )
    plre.ResourceMultiControlledX(2+ctrlqubits, 2+ctrlzeros, wires=list(range(systq+beq+qspq,systq+beq+qspq+ctrlqubits))+list(range(0, qspq+beq)))
    plre.ResourceControlled(plre.ResourceHadamard(wires=[1]) , num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) )

    ##LAST LCU STEP
    plre.ResourceControlled(plre.ResourceHadamard(wires=[0]), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq+beq+qspq,systq+beq+qspq+ctrlqubits) ) 

    return

def CTRL_FFTBASED_CIRCUIT(systq,  ctrlqubits=1, ctrlzeros=0):
    # plre.ResourceControlled(plre.ResourceQFT(systq, wires=range(0, systq)), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq,systq+ctrlqubits))
    plre.ResourceQFT(systq, wires=range(0, systq))

    ####CONTROLLED ROTATIONS
    for j in range(systq):
        plre.ResourceControlled(plre.ResourceRY(wires=j), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq,systq+ctrlqubits))

    # plre.ResourceControlled(plre.ResourceQFT(systq, wires=range(0, systq)), num_ctrl_wires=ctrlqubits, num_ctrl_values=ctrlzeros, wires=range(systq,systq+ctrlqubits))
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

def PARAMETERIZED_STATE_PREP(systq, Ns):
    for k in range(Ns):
        ###call ep^2
        plre.ResourceQFT(wires=range(systq), num_wires=systq)
        for j in range(systq):
            plre.ResourcePhaseShift(wires=[j])
        for j in range(int(systq*(systq-1)/2)): ##extremly lasy strategy for the controlled phase gates
            plre.ResourceControlledPhaseShift()
        plre.ResourceQFT(num_wires=systq, wires=range(systq))
        
        ##call ex^2
        for k in range(systq):
            plre.ResourcePhaseShift(wires=[k])
        
        for j in range(int(systq*(systq-1)/2)): ##extremly lazy strategy for the controlled phase gates
            plre.ResourceControlledPhaseShift()

        for l in range(systq):
            plre.ResourceRZ(wires=[l])
            plre.ResourceRX(wires=[l])
            plre.ResourceRY(wires=[l])

    return

my_gate_set = {"Hadamard", "S", "T", "CNOT", "Toffoli"}
# QSP_NONHERMITIAN_LCU_CIRCUIT(4, 3)
# print(plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(6, ctrlqubits=14))
# print(plre.estimate(CTRL_QSP_NONHERMITIAN_LCU_CIRCUIT, my_gate_set)(22, 6, ctrlqubits=14))
# print(plre.estimate(QAA_QSP_NONHERMITIAN_LCU_CIRCUIT, my_gate_set)(4, 20, 3 ))

# A=plre.estimate(QAA_QSP_NONHERMITIAN_LCU_CIRCUIT, my_gate_set)(4, 20, 3 )
# print(A.clean_gate_counts["T"])

truncatedspacequbits=np.array([6])
# truncatedspacequbits=np.array([2])
Tcountsqsp=[]
Tcountsparam=[]
eregisterqubitsrange=[5, 6, 7, 8, 9, 10]
def binary(num, length=4):
    return format(num, '#0{}b'.format(length + 2)).replace('0b', '')

value=truncatedspacequbits[0]

#####QSP BASED ESTIMATE####
for ind, eregisterqubits in enumerate(eregisterqubitsrange):
# for ind, value in enumerate(truncatedspacequbits):
    qspres=plre.estimate(QAA_QSP_NONHERMITIAN_LCU_CIRCUIT, my_gate_set)(4, 22, value)
    # QSpcircstotal=(2**eregisterqubits)*value#.clean_gate_counts["T"]
    qffres=plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(0,  ctrlqubits=eregisterqubits, ctrlzeros=eregisterqubits)
    
    for j in range(1, 2**eregisterqubits):
        zerocount=binary(j, eregisterqubits).count('0')
        qffres+=plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(value,  ctrlqubits=eregisterqubits, ctrlzeros=zerocount)
    totalcount=(2**eregisterqubits)*qspres+qffres
    Tcountsqsp.append(totalcount.clean_gate_counts["T"])

#PARAMETERIZED ESTIMATE###
for ind, eregisterqubits in enumerate(eregisterqubitsrange):
# for ind, value in enumerate(truncatedspacequbits):
    qspres=plre.estimate(PARAMETERIZED_STATE_PREP, my_gate_set)(value, 22)
    # QSpcircstotal=(2**eregisterqubits)*value#.clean_gate_counts["T"]
    qffres=plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(0,  ctrlqubits=eregisterqubits, ctrlzeros=eregisterqubits)
    
    for j in range(1, 2**eregisterqubits):
        zerocount=binary(j, eregisterqubits).count('0')
        qffres+=plre.estimate(CTRL_FFTBASED_CIRCUIT, my_gate_set)(value,  ctrlqubits=eregisterqubits, ctrlzeros=zerocount)
    totalcount=(2**eregisterqubits)*qspres+qffres
    Tcountsparam.append(totalcount.clean_gate_counts["T"])

# plt.plot(eregisterqubitsrange, np.log(Tcountsqsp), marker='x', label='qsp"')

plt.plot(eregisterqubitsrange,(Tcountsparam), marker='x', label="param")
# plt.xlabel("Hilbert space truncation, log(N)")
plt.xlabel("Log of number of sites")
# plt.ylabel("log(T Count)")
plt.ylabel("T count")

pathname="pennylanecircuits.py"
current_path=os.path.abspath(__file__)
coeff_path=current_path.replace(pathname, "")
save_path = os.path.normpath(coeff_path)
tikzplotlib.save(os.path.join(save_path, "comp_vqa_22_logscale.tex"), flavor="context")

# tikzplotlib.save(os.path.join(save_path, "comp_qsp_vqa_22_logscale.tex"), flavor="context")

plt.legend()
plt.show()