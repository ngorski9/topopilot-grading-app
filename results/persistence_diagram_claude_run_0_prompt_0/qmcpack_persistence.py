from paraview.simple import *
from paraview import servermanager as sm
import vtk.util.numpy_support as vnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARRAY = "Scalars_"
THRESHOLD = 0.04
INPUT_FILE = "/workspace/QMCPACK.vti"
OUTPUT_PNG = "/workspace/QMCPACK_persistence_diagram.png"
OUTPUT_VTP = "/workspace/QMCPACK_persistence_diagram_orders0_2.vtp"

reader = XMLImageDataReader(FileName=[INPUT_FILE])
reader.PointArrayStatus = [ARRAY]

# 1) Persistence diagram of the raw field -> drives the simplification
pd0 = TTKPersistenceDiagram(Input=reader)
pd0.ScalarField = ['POINTS', ARRAY]
pd0.EmbedinDomain = 0

# 2) Keep only persistent pairs (> threshold) as simplification constraints
critPairs = Threshold(Input=pd0)
critPairs.Scalars = ['CELLS', 'Persistence']
critPairs.LowerThreshold = THRESHOLD
critPairs.UpperThreshold = 1e9
critPairs.ThresholdMethod = 'Between'

# 3) Topologically simplify the scalar field
simplify = TTKTopologicalSimplification(Domain=reader, Constraints=critPairs)
simplify.ScalarField = ['POINTS', ARRAY]
simplify.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']

# 4) Recompute the persistence diagram on the simplified field
pdFinal = TTKPersistenceDiagram(Input=simplify)
pdFinal.ScalarField = ['POINTS', ARRAY]
pdFinal.EmbedinDomain = 0
pdFinal.UpdatePipeline()

data = sm.Fetch(pdFinal)
if data.IsA('vtkMultiBlockDataSet'):
    it = data.NewIterator()
    it.InitTraversal()
    block = it.GetCurrentData()
else:
    block = data

cd = block.GetCellData()
birth = vnp.vtk_to_numpy(cd.GetArray('Birth'))
persistence = vnp.vtk_to_numpy(cd.GetArray('Persistence'))
death = birth + persistence
pairType = vnp.vtk_to_numpy(cd.GetArray('PairType'))

mask02 = (pairType == 0) | (pairType == 2)
print(f"Total pairs: {len(birth)}, orders {{0,2}} pairs: {mask02.sum()}")

fig, ax = plt.subplots(figsize=(7, 7))
colors = {0: '#1f77b4', 2: '#d62728'}
labels = {0: 'Order 0 (minimum-saddle)', 2: 'Order 2 (saddle-maximum)'}
for order in (0, 2):
    m = pairType == order
    ax.scatter(birth[m], death[m], s=18, c=colors[order], label=labels[order], alpha=0.8)

lo = min(birth.min(), death.min())
hi = max(birth.max(), death.max())
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='diagonal (birth = death)')
ax.set_xlabel('Birth')
ax.set_ylabel('Death')
ax.set_title(f'QMCPACK sublevel-set persistence diagram\n(simplified, threshold={THRESHOLD})')
ax.legend(loc='upper left', fontsize=8)
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=150)
print(f"Saved persistence diagram image to {OUTPUT_PNG}")

order02 = Threshold(Input=pdFinal)
order02.Scalars = ['CELLS', 'PairType']
order02.LowerThreshold = 0
order02.UpperThreshold = 2
order02.ThresholdMethod = 'Between'
order02.AllScalars = 0

pt1 = Threshold(Input=order02)
pt1.Scalars = ['CELLS', 'PairType']
pt1.LowerThreshold = 1
pt1.UpperThreshold = 1
pt1.ThresholdMethod = 'Between'
pt1.Invert = 1

merged02 = MergeBlocks(Input=pt1)
merged02.UpdatePipeline()
SaveData(OUTPUT_VTP.replace('.vtp', '.vtu'), proxy=merged02)
print(f"Saved filtered persistence pairs (orders 0 and 2) to {OUTPUT_VTP}")
