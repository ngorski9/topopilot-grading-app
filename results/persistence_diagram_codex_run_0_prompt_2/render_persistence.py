from paraview.simple import *
from paraview import servermanager
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LoadPlugin('TopologyToolKit', remote=False, ns=globals())
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.PointArrayStatus = ['Scalars_']
simp = TTKTopologicalSimplificationByPersistence(Input=reader)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PairType = 'Extremum-Saddle'
simp.PersistenceThreshold = 0.04
simp.ThresholdIsAbsolute = 0

pd = TTKPersistenceDiagram(Input=simp)
pd.ScalarField = ['POINTS', 'Scalars_']
pd.Dimensions = 'All Dimensions'
pd.ThreadNumber = 8
pd.UseAllCores = 1
SaveState('/workspace/QMCPACK_persistence_diagram_pipeline.pvsm')

UpdatePipeline(proxy=pd)

data = servermanager.Fetch(pd)
print('cells',data.GetNumberOfCells(),'points',data.GetNumberOfPoints())
for i in range(data.GetCellData().GetNumberOfArrays()):
 a=data.GetCellData().GetArray(i); print('cell',a.GetName(),a.GetNumberOfTuples(),a.GetRange())
for i in range(data.GetPointData().GetNumberOfArrays()):
 a=data.GetPointData().GetArray(i); print('point',a.GetName(),a.GetNumberOfTuples(),a.GetRange())

# The persistence-diagram geometry stores each pair as a vertical segment at
# (birth, death); PairType 0 and 2 correspond to H0 and H2 respectively.
pairs = {0: [], 2: []}
types = data.GetCellData().GetArray('PairType')
finite = data.GetCellData().GetArray('IsFinite')
for i in range(data.GetNumberOfCells()):
    dim = int(types.GetTuple1(i))
    if dim not in pairs:
        continue
    cell = data.GetCell(i)
    pts = [data.GetPoint(cell.GetPointId(j)) for j in range(cell.GetNumberOfPoints())]
    birth, death = pts[0][0], pts[-1][1]
    pairs[dim].append((birth, death, int(finite.GetTuple1(i))))

fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
allvals = [v for values in pairs.values() for pair in values for v in pair[:2]]
lo, hi = min(allvals), max(allvals)
pad = max((hi - lo) * 0.06, 0.03)
ax.fill_between([lo-pad, hi+pad], [lo-pad, hi+pad], hi+pad, color='#f3f4f6', zorder=0)
ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], '--', color='#64748b', lw=1.2, label='diagonal')
styles = {0: ('#2563eb', 'H$_0$'), 2: ('#dc2626', 'H$_2$')}
for dim, values in pairs.items():
    color, label = styles[dim]
    finite_pairs = [(b,d) for b,d,f in values if f]
    essential_pairs = [(b,d) for b,d,f in values if not f]
    if finite_pairs:
        ax.scatter(*zip(*finite_pairs), s=42, c=color, edgecolors='white', linewidths=.65, label=label, zorder=3)
    if essential_pairs:
        ax.scatter(*zip(*essential_pairs), s=70, c=color, marker='D', edgecolors='black', linewidths=.65, label=f'{label} essential', zorder=4)
ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('Birth'); ax.set_ylabel('Death')
ax.set_title('Piecewise-linear Sub-level Set Persistence Diagram\nQMCPACK — persistence simplification threshold 0.04')
ax.grid(True, color='#e5e7eb', linewidth=.7)
ax.legend(loc='upper left', frameon=True)
fig.tight_layout()
fig.savefig('/workspace/QMCPACK_persistence_diagram_H0_H2_threshold_0.04.png', facecolor='white')
print('H0 pairs', len(pairs[0]), 'H2 pairs', len(pairs[2]))
