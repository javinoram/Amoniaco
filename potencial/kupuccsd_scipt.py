
from base import *
import pandas as pd
import vqesimulation as qs

distance = np.linspace(1, 8, 31)
learning_rate = [0.01, 0.1, 0.2, 0.3, 0.4]
Phase_energy = []
Phase_optimum = []

for lr in learning_rate:
    minimizate_params["theta"] = ["adam", lr]
    minimizate_params["number"] = a*b
    Phase_energy = []
    Phase_optimum = []

    for d in distance:
        coordinates = space(d)
        system = qs.vqe_molecular(symbols, coordinates, mol_params)

        ansazt = qs.kupccgsd_ansatz()
        ansazt.set_device( ansatz_params )
        ansazt.set_node( ansatz_params )
        ansazt.set_state( ansatz_params["electrons"], sz=0 )

        system.set_node(ansazt.node, ansatz_params["interface"])
        optimizer = qs.gradiend_optimizer(minimizate_params)
        optimizer.get_energy = True
        optimizer.get_params = True

        energy, optimum = optimizer.VQE(system.cost_function)

        Phase_optimum.append( optimum )
        Phase_energy.append( energy )

    Phase_energy = pd.DataFrame(Phase_energy)
    Phase_optimum = pd.DataFrame(Phase_optimum)

    Phase_energy.to_csv("datos/potencial/Energykupccgsd"+str(lr)+"-"+str(d)+".csv")
    Phase_optimum.to_csv("datos/potencial/Thetakupccgsd"+str(lr)+"-"+str(d)+".csv")