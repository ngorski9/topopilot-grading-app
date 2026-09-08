"""TTK workflow: simplify QMCPACK scalar field, then plot H0 and H2 pairs."""
from paraview.simple import *
from paraview import servermanager
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter, vtkXMLUnstructuredGridWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = 'QMCPACK.vti'
THRESHOLD = 0.04
reader = XMLImageDataReader(FileName=[INPUT])
simplified = TTKTopologicalSimplificationByPersistence(Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = THRESHOLD
simplified.ThresholdIsAbsolute = 1
diagram = TTKPersistenceDiagram(Input=simplified)
diagram.ScalarField = ['POINTS', 'Scalars_']
# Compute all dimensions; the exported visualization filters PairType to 0 and 2.
UpdatePipeline(proxy=diagram)

simp_data = servermanager.Fetch(simplified)
pd_data = servermanager.Fetch(diagram)
vti = vtkXMLImageDataWriter(); vti.SetFileName('QMCPACK_simplified_persistence_0.04.vti'); vti.SetInputData(simp_data); vti.Write()
vtu = vtkXMLUnstructuredGridWriter(); vtu.SetFileName('QMCPACK_persistence_diagram_simplified_0.04.vtu'); vtu.SetInputData(pd_data); vtu.Write()

pair_type = pd_data.GetCellData().GetArray('PairType')
persistence = pd_data.GetCellData().GetArray('Persistence')
orders = {0: ('tab:blue', 'Order 0'), 2: ('tab:orange', 'Order 2')}
points = {0: [], 2: []}
for cell_id in range(pd_data.GetNumberOfCells()):
    order = int(pair_type.GetTuple1(cell_id))
    if order not in orders:
        continue
    ids = pd_data.GetCell(cell_id).GetPointIds()
    a, b = pd_data.GetPoint(ids.GetId(0)), pd_data.GetPoint(ids.GetId(1))
    points[order].append((a[0], b[1], persistence.GetTuple1(cell_id)))

fig, ax = plt.subplots(figsize=(8.5, 7.2), dpi=180)
all_values = [v for values in points.values() for p in values for v in p[:2]]
lo, hi = min(all_values), max(all_values)
pad = max((hi - lo) * 0.05, 0.02)
ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color='0.35', lw=1.2, ls='--', label='Diagonal')
for order, (color, label) in orders.items():
    vals = points[order]
    if vals:
        ax.scatter([p[0] for p in vals], [p[1] for p in vals], s=34, color=color, edgecolors='white', linewidths=.45, alpha=.92, label=f'{label} (n={len(vals)})')
ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad); ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Birth'); ax.set_ylabel('Death')
ax.set_title('QMCPACK persistence diagram — sublevel-set filtration\nTTK persistence simplification (absolute threshold = 0.04)')
ax.grid(True, alpha=.22); ax.legend(loc='upper left', frameon=True)
fig.tight_layout()
fig.savefig('QMCPACK_persistence_diagram_orders_0_2_threshold_0.04.png', bbox_inches='tight')
print('H0 pairs:', len(points[0]))
print('H2 pairs:', len(points[2]))
print('Wrote simplified VTI, persistence-diagram VTU, and PNG.')
