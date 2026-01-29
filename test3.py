import time
import numpy as np
from openep.io_utils import readers, writers
from openep._datasets.openep_datasets import DATASET_2, DATASET_1

import pyvista as pv
from openep.mesh_tools.newregistration import CPDRegistrationWorker

case2 = readers.load_openep_mat(DATASET_2)
case1 = readers.load_openep_mat(DATASET_1)

#print(case1.points)
#print(case2.points)



cpd = CPDRegistrationWorker({'pars': {'method': 'affine', 'kwargs': {}}})
res_64 = cpd.run(case1.points, case2.points)
print(res_64)


pl = pv.Plotter()
pl.add_points(case1.points, color='red')
pl.add_points(case2.points, color='blue')
pl.add_points(res_64, color='green')
pl.show()

case3 = writers.export_openep_mat({'points': res_64}, 'registered_case.mat')



