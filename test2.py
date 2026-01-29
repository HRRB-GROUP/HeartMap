from openep.io_utils import readers
from openep._datasets.openep_datasets import DATASET_2, DATASET_1
import pyvista as pv
case1 = readers.load_openep_mat(DATASET_1)
case2= readers.load_openep_mat(DATASET_2)
pv.PolyData(case1.points)
pl = pv.Plotter()
pl.add_points(case1.points, color='red')
pl.add_points(case2.points, color='blue')
pl.show()
