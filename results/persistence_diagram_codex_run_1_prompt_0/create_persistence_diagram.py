"""Create an order-0/order-2 persistence diagram after TTK simplification."""
from paraview.simple import (
    XMLImageDataReader,
    TTKTopologicalSimplificationByPersistence,
    TTKPersistenceDiagram,
    SaveData,
)
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter
from vtkmodules.vtkCommonCore import vtkIdTypeArray
from vtkmodules.vtkCommonDataModel import vtkSelectionNode, vtkSelection
from vtkmodules.vtkFiltersExtraction import vtkExtractSelection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = "/workspace/QMCPACK.vti"
OUT = "/workspace/output"

# Persistence simplification is performed directly on the scalar field.
source = XMLImageDataReader(FileName=[INPUT])
simplified = TTKTopologicalSimplificationByPersistence(Input=source)
simplified.InputArray = ["POINTS", "Scalars_"]
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1
simplified.DebugLevel = 0
simplified.UpdatePipeline()
SaveData(f"{OUT}/QMCPACK_simplified_threshold_0.04.vti", proxy=simplified)

# Compute the sublevel-set persistence diagram of the simplified field.
diagram = TTKPersistenceDiagram(Input=simplified)
diagram.ScalarField = ["POINTS", "Scalars_"]
diagram.Dimensions = "All Dimensions"
diagram.DebugLevel = 0
diagram.UpdatePipeline()
SaveData(f"{OUT}/persistence_diagram_all_orders.vtu", proxy=diagram)

# Extract only orders 0 and 2 into a reusable VTK dataset.
reader = vtkXMLUnstructuredGridReader()
reader.SetFileName(f"{OUT}/persistence_diagram_all_orders.vtu")
reader.Update()
data = reader.GetOutput()
pair_type = data.GetCellData().GetArray("PairType")
ids = vtkIdTypeArray()
for cell_id in range(data.GetNumberOfCells()):
    if int(pair_type.GetTuple1(cell_id)) in (0, 2):
        ids.InsertNextValue(cell_id)
node = vtkSelectionNode()
node.SetFieldType(vtkSelectionNode.CELL)
node.SetContentType(vtkSelectionNode.INDICES)
node.SetSelectionList(ids)
selection = vtkSelection()
selection.AddNode(node)
extract = vtkExtractSelection()
extract.SetInputData(0, data)
extract.SetInputData(1, selection)
extract.Update()
writer = vtkXMLUnstructuredGridWriter()
writer.SetFileName(f"{OUT}/persistence_diagram_orders_0_2.vtu")
writer.SetInputData(extract.GetOutput())
writer.Write()

# A conventional persistence diagram uses x=birth and y=death (top endpoint).
birth = data.GetCellData().GetArray("Birth")
persistence = data.GetCellData().GetArray("Persistence")
points = {0: ([], []), 2: ([], [])}
for cell_id in range(data.GetNumberOfCells()):
    order = int(pair_type.GetTuple1(cell_id))
    if order in points:
        b = birth.GetTuple1(cell_id)
        points[order][0].append(b)
        points[order][1].append(b + persistence.GetTuple1(cell_id))

max_value = max(y for _, ys in points.values() for y in ys)
fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
ax.plot([0, max_value], [0, max_value], "--", color="0.45", lw=1.2, label="diagonal")
ax.scatter(*points[0], s=55, marker="o", color="#1774b7", label="Order 0", zorder=3)
ax.scatter(*points[2], s=70, marker="^", color="#d1495b", label="Order 2", zorder=3)
ax.set(xlim=(0, max_value * 1.03), ylim=(0, max_value * 1.03), aspect="equal",
       xlabel="Birth", ylabel="Death",
       title="QMCPACK — sublevel-set persistence diagram\nPersistence simplification (absolute threshold = 0.04)")
ax.grid(True, alpha=0.25)
ax.legend(loc="upper left")
fig.savefig(f"{OUT}/persistence_diagram_orders_0_2.png", dpi=200)
print(f"order-0 pairs: {len(points[0][0])}; order-2 pairs: {len(points[2][0])}")
