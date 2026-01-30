from openep.io_utils import readers
from openep._datasets.openep_datasets import DATASET_1, DATASET_2
import pyvista as pv
from openep.mesh_tools.registration2 import CPDReconstructionCase
from openep.mesh_tools.register1 import CPDRegistrationPoints

# Load cases
case_source = readers.load_openep_mat(DATASET_1)
case_target = readers.load_openep_mat(DATASET_2)

# Registration (points only)
cpd = CPDRegistrationPoints(method="affine")
registered_points = cpd.run(case_source.points, case_target.points)

# Reconstruction
reconstructor = CPDReconstructionCase(case_source)
registered_case = reconstructor.build_registered_case(
    registered_points,
    output_path="registered_case.mat"
)

# Visualisation
pl = pv.Plotter()
pl.add_mesh(pv.PolyData(case_target.points, case_target.indices), color="blue")
pl.add_mesh(pv.PolyData(registered_points, case_source.indices), color="green")
pl.show()
