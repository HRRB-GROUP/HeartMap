import time
import numpy as np

current_time = time.time()
from openep.io_utils import readers
elapsed_time = time.time() - current_time
print(f"Importing readers took {elapsed_time:.4f} seconds")

current_time = time.time()
from openep._datasets.openep_datasets import DATASET_2, DATASET_1
elapsed_time = time.time() - current_time
print(f"Importing datasets took {elapsed_time:.4f} seconds")

current_time = time.time()
from openep.mesh_tools.cpd_core import CPDRegistrationWorker
elapsed_time = time.time() - current_time
print(f"Importing CPDRegistrationWorker took {elapsed_time:.4f} seconds")

case2 = readers.load_openep_mat(DATASET_2)
case1 = readers.load_openep_mat(DATASET_1)

#print(case1.points)
#print(case2.points)

cpd=CPDRegistrationWorker({'pars': {
    'method': 'rigid',
    'n_steps': 5,
    'kwargs': {}
}})

#cpd.run(case2.points, case1.points)


cpd_64 = CPDRegistrationWorker({'pars': {'method': 'affine', 'n_steps': 5, 'kwargs': {}}})
res_64 = cpd_64.run(case1.points.astype(np.float64), case2.points.astype(np.float64))


cpd_32 = CPDRegistrationWorker({'pars': {'method': 'affine', 'n_steps': 5, 'kwargs': {}}})
res_32 = cpd_32.run(case1.points.astype(np.float32), case2.points.astype(np.float32))


diff = res_64 - res_32.astype(np.float64)
distances = np.linalg.norm(diff, axis=1)

print(f"Mean error : {np.mean(distances):.6e} units")
print(f"Max error : {np.max(distances):.6e} units")