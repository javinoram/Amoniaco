import numpy as np
import pennylane as qml
import sys

sys.path.append("../vqesimulation")
import quantumsim as qs

def get_state_UCCSD(angle, ansatz_params):
    ansatz_params["diff_method"] = "best"

    ansazt = qs.uccds_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_exitations( ansatz_params["electrons"], sz=0 )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )
    return np.round( np.real( ansazt.get_state( angle ) ), 7)

def get_state_kUpCCGSD(angle, ansatz_params):
    ansatz_params["diff_method"] = "best"

    ansazt = qs.kupccgsd_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )
    return np.round( np.real( ansazt.get_state( angle ) ), 7)

def space(x):
    x1 = [0.0, 0.0, x*(3/10)*(0.375324205)]
    x2 = [0.0, 2*x*(0.4634468), -x*(7/10)*(0.375324205)]
    x3 = [-x*np.sin(0.9318313), -x*0.4634468, -x*(7/10)*(0.375324205)]
    x4 = [x*np.sin(0.9318313), -x*0.4634468, -x*(7/10)*(0.375324205)]
    final = []
    final.extend(x1)
    final.extend(x2)
    final.extend(x3)
    final.extend(x4)
    return np.array(final)


symbols = ["N", "H", "H", "H"]
mol_params = {
    'mapping': "jordan_wigner",
    'charge': 0, 
    'mult': 1,
    'basis': 'sto-3g',
    'method': 'dhf',
    'active_electrons': 8,
    'active_orbitals': 7,
    }

ansatz_params = {
    "repetitions": 1,
    "base": "lightning.qubit",
    "interface": "autograd",
    "electrons": 8,
    "qubits": mol_params["active_orbitals"]*2,
    "diff_method": "adjoint"
}

a,b = qml.kUpCCGSD.shape(k=ansatz_params["repetitions"], n_wires=ansatz_params["qubits"], delta_sz=0)
singles, doubles = qml.qchem.excitations(mol_params["active_electrons"], mol_params["active_orbitals"]*2, 0)
singles = len(singles)
doubles = len(doubles)

minimizate_params = {
    "maxiter": 1500,
    "tol": 1e-6,
    "number": 0,
    "theta":["adam", 0.2]}