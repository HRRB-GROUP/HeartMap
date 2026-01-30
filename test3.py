import time
import numpy as np
from openep.io_utils import readers, writers
from openep._datasets.openep_datasets import DATASET_2, DATASET_1, DATASET_registered

import pyvista as pv
from openep.mesh_tools.newregistration import CPDRegistrationWorker

import pyvista as pv


def case_to_pyvista_mesh(case):
    faces = np.hstack(
        [np.full((case.indices.shape[0], 1), 3), case.indices]
    ).astype(np.int64)

    return pv.PolyData(case.points, faces)


case2 = readers.load_openep_mat(DATASET_2)
case1 = readers.load_openep_mat(DATASET_1)
case_registered = readers.load_openep_mat(DATASET_registered)

#print(case1.points)
#print(case2.points)



#cpd = CPDRegistrationWorker({'pars': {'method': 'affine', 'kwargs': {}}})
#res_64 = cpd.run(case1.points, case2.points)
#print(res_64)
def case_to_pyvista_mesh(case):
    faces = np.hstack(
        [np.full((case.indices.shape[0], 1), 3), case.indices]
    ).astype(np.int64)

    return pv.PolyData(case.points, faces)

mesh = case_to_pyvista_mesh(case_registered)
mesh["bipolar_voltage"] = case_registered.fields["bipolar_voltage"]



pl = pv.Plotter()

pl.add_mesh(case_to_pyvista_mesh(case1), color='red', opacity=0.5)
pl.add_mesh(case_to_pyvista_mesh(case2), color='blue', opacity=0.5)
pl.add_mesh(case_to_pyvista_mesh(case_registered), color='green')
#pl.add_mesh(mesh, scalars="bipolar_voltage", cmap="viridis")

pl.show()



