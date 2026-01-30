from openep.io_utils import readers, writers
from openep._datasets.openep_datasets import DATASET_1, DATASET_2
import pyvista as pv
from openep.mesh_tools.registration2 import CPDReconstructedCase
from openep.mesh_tools.register1 import CPDRegistrationPoints

# Load
case_src = readers.load_openep_mat(DATASET_1)
case_tgt = readers.load_openep_mat(DATASET_2)

# Register
cpd = CPDRegistrationPoints(method="affine")
registered_points = cpd.run(case_src.points, case_tgt.points)

# Rebuild Case
reconstructor = CPDReconstructedCase(case_src)
registered_case = reconstructor.build(registered_points)

# Export 
writers.export_openep_mat(registered_case, "registered_case.mat")


# Plot 
pl = pv.Plotter()
pl.add_mesh(pv.PolyData(case_tgt.points, case_tgt.indices), color="blue")
pl.add_mesh(pv.PolyData(registered_case.points, registered_case.indices), color="green")
pl.show()
