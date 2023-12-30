
from base import *
import pandas as pd
import vqesimulation as qs

distance = 1.5
learning_rate = [0.01, 0.1, 0.2, 0.3, 0.4]
Phase_energy = []
Phase_optimum = []

for lr in learning_rate:
    minimizate_params["theta"] = ["adam", lr]
    minimizate_params["number"] = a*b
    Phase_energy = []
    Phase_optimum = []

    coordinates = space(distance)
    system = qs.structure_molecular(symbols, coordinates, mol_params)

    ansazt = qs.kupccgsd_ansatz()
    ansazt.set_device( ansatz_params )
    ansazt.set_node( ansatz_params )
    ansazt.set_state( ansatz_params["electrons"], sz=0 )

    system.set_node(ansazt.node, ansatz_params["interface"])
    optimizer = qs.gradiend_optimizer(minimizate_params)
    optimizer.get_energy = True
    optimizer.get_params = True

    energy, optimum = optimizer.OS(system.cost_function, coordinates, system.grad_x)

    Phase_optimum.append( optimum )
    Phase_energy.append( energy )

Phase_energy = pd.DataFrame(Phase_energy)
Phase_optimum = pd.DataFrame(Phase_optimum)

Phase_energy.to_csv("datos/relajacion/Energykupccgsd"+str(lr)+"-"+str(distance)+".csv")
Phase_optimum.to_csv("datos/relajacion/Thetakupccgsd"+str(lr)+"-"+str(distance)+".csv")