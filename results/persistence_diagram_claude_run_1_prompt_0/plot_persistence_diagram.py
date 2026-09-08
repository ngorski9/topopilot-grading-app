from paraview.simple import *
from paraview.servermanager import Fetch
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LoadPlugin('TopologyToolKit', remote=False, ns=globals())

reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

pd0 = TTKPersistenceDiagram(Input=reader)
pd0.ScalarField = ['POINTS', 'Scalars_']
pd0.UpdatePipeline()

thresh = Threshold(Input=pd0)
thresh.Scalars = ['CELLS', 'Persistence']
thresh.LowerThreshold = 0.04
thresh.UpperThreshold = 1e18
thresh.ThresholdMethod = 'Between'
thresh.UpdatePipeline()

simplify = TTKTopologicalSimplification(Domain=reader, Constraints=thresh)
simplify.ScalarField = ['POINTS', 'Scalars_']
simplify.UpdatePipeline()

pd1 = TTKPersistenceDiagram(Input=simplify)
pd1.ScalarField = ['POINTS', 'Scalars_']
pd1.UpdatePipeline()

data = Fetch(pd1)
if data.IsA('vtkMultiBlockDataSet'):
    it = data.NewIterator()
    it.InitTraversal()
    ug = it.GetCurrentDataObject()
else:
    ug = data

pts = vtk_to_numpy(ug.GetPoints().GetData())
cd = ug.GetCellData()
pairtype = vtk_to_numpy(cd.GetArray('PairType'))
persistence = vtk_to_numpy(cd.GetArray('Persistence'))

ncells = ug.GetNumberOfCells()
births = np.zeros(ncells)
deaths = np.zeros(ncells)
for i in range(ncells):
    cell = ug.GetCell(i)
    p0 = cell.GetPointId(0)
    p1 = cell.GetPointId(1)
    births[i] = pts[p0][0]
    deaths[i] = pts[p1][1]

# PairType: 0 = minimum-saddle (dim 0), 1 = saddle-saddle (dim 1, only in 3D), 2 = saddle-maximum (dim 2)
# -1 marks the global min-max pair (infinite persistence pair), treat as dim 0 (connected component)
dim0_mask = (pairtype == 0) | (pairtype == -1)
dim2_mask = (pairtype == 2)

print(f"Total pairs: {ncells}")
print(f"Dim-0 pairs (minima-saddle): {dim0_mask.sum()}")
print(f"Dim-2 pairs (saddle-maxima): {dim2_mask.sum()}")

fig, ax = plt.subplots(figsize=(7, 7))

all_vals = np.concatenate([births, deaths])
lo, hi = all_vals.min(), all_vals.max()
pad = (hi - lo) * 0.05
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', linewidth=1, label='diagonal (birth = death)')

ax.scatter(births[dim0_mask], deaths[dim0_mask], c='tab:blue', s=25, label='Order 0 (minima-saddle)', alpha=0.8)
ax.scatter(births[dim2_mask], deaths[dim2_mask], c='tab:red', s=25, marker='^', label='Order 2 (saddle-maxima)', alpha=0.8)

ax.set_xlabel('Birth')
ax.set_ylabel('Death')
ax.set_title('QMCPACK.vti Persistence Diagram\n(Sublevel-set filtration, simplified at threshold=0.04)')
ax.legend(loc='lower right')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(lo - pad, hi + pad)

plt.tight_layout()
plt.savefig('/workspace/QMCPACK_persistence_diagram.png', dpi=150)
print("Saved plot to /workspace/QMCPACK_persistence_diagram.png")
