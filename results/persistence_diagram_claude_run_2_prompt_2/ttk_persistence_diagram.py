from paraview.simple import *
import paraview

paraview.compatibility.major = 5
paraview.compatibility.minor = 13

INPUT_FILE = "/workspace/QMCPACK.vti"
ARRAY_NAME = "Scalars_"
SIMPLIFICATION_THRESHOLD = 0.04

# 1. Read the scalar field
reader = XMLImageDataReader(FileName=[INPUT_FILE])
reader.PointArrayStatus = [ARRAY_NAME]

# 2. Initial persistence diagram (sub-level set / ascending filtration) used
#    to drive topological simplification.
initialDiagram = TTKPersistenceDiagram(Input=reader)
initialDiagram.ScalarField = ['POINTS', ARRAY_NAME]
initialDiagram.InputOffsetField = ['POINTS', ARRAY_NAME]
initialDiagram.EmbedinDomain = 0
initialDiagram.UpdatePipeline()

# 3. Simplify the scalar field: keep only critical point pairs with
#    persistence above the requested threshold.
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=initialDiagram)
simplification.ScalarField = ['POINTS', ARRAY_NAME]
simplification.InputOffsetField = ['POINTS', ARRAY_NAME]
simplification.Backend = 'LTS (IEEE VIS 2020)'
simplification.ThresholdMethod = 'Persistence'
simplification.Threshold = SIMPLIFICATION_THRESHOLD

# 4. Piecewise-linear sub-level set filtration persistence diagram of the
#    simplified field.
finalDiagram = TTKPersistenceDiagram(Input=simplification)
finalDiagram.ScalarField = ['POINTS', ARRAY_NAME]
finalDiagram.InputOffsetField = ['POINTS', ARRAY_NAME]
finalDiagram.EmbedinDomain = 0
finalDiagram.UpdatePipeline()

# 5. Keep only H0 (min-saddle) and H2 (saddle-max) pairs.
#    TTK encodes pair type in the "PairType" cell array: 0 = min-saddle (H0),
#    1 = saddle-saddle (H1, only in 3D), 2 = saddle-max (H2).
h0h2 = Threshold(Input=finalDiagram)
h0h2.Scalars = ['CELLS', 'PairType']
h0h2.ThresholdMethod = 'Between'
h0h2.LowerThreshold = -0.5
h0h2.UpperThreshold = 0.5

h2 = Threshold(Input=finalDiagram)
h2.Scalars = ['CELLS', 'PairType']
h2.ThresholdMethod = 'Between'
h2.LowerThreshold = 1.5
h2.UpperThreshold = 2.5

merged = AppendDatasets(Input=[h0h2, h2])
merged.UpdatePipeline()

SaveData('/workspace/persistence_diagram_h0_h2.vtu', proxy=merged)
SaveData('/workspace/persistence_diagram_full.vtu', proxy=finalDiagram)

# 6. Render a proper 2D birth/death persistence diagram (H0 and H2 only)
#    using matplotlib, reading the coordinates straight back from the VTU
#    just written (x = birth, y = death, since EmbedinDomain = 0).
import vtk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

vtu_reader = vtk.vtkXMLUnstructuredGridReader()
vtu_reader.SetFileName('/workspace/persistence_diagram_h0_h2.vtu')
vtu_reader.Update()
data = vtu_reader.GetOutput()

pair_type_arr = data.GetCellData().GetArray('PairType')
points = data.GetPoints()

fig, ax = plt.subplots(figsize=(8, 8))
colors = {0: ('#1f77b4', 'H0 (min-saddle)'), 2: ('#d62728', 'H2 (saddle-max)')}
plotted_labels = set()

for c in range(data.GetNumberOfCells()):
    ptype = int(pair_type_arr.GetValue(c))
    if ptype not in colors:
        continue
    cell = data.GetCell(c)
    p0 = points.GetPoint(cell.GetPointId(0))
    p1 = points.GetPoint(cell.GetPointId(1))
    birth, death = sorted([p0[0], p1[0]])  # x holds birth/death scalar pair
    color, label = colors[ptype]
    lbl = label if label not in plotted_labels else None
    plotted_labels.add(label)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], marker='o', color=color, label=lbl)

lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='diagonal (birth = death)')

ax.set_xlabel('Birth')
ax.set_ylabel('Death')
ax.set_title('Persistence Diagram (H0 & H2), simplification = 0.04\nPiecewise-linear sub-level set filtration')
ax.legend()
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
fig.savefig('/workspace/persistence_diagram_h0_h2.png', dpi=150)

print("Done. Outputs:")
print(" /workspace/persistence_diagram_h0_h2.vtu")
print(" /workspace/persistence_diagram_full.vtu")
print(" /workspace/persistence_diagram_h0_h2.png")
