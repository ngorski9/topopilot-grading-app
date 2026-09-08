from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = "/workspace/QMCPACK.vti"
OUT = "/workspace/QMCPACK_persistence_diagram_orders_0_2.png"

# Compute a simplified scalar field: pairs with absolute persistence < 0.04
# are cancelled before calculating the final sublevel-set persistence diagram.
source = XMLImageDataReader(FileName=[INPUT])
source.PointArrayStatus = ["Scalars_"]
simplified = TTKTopologicalSimplificationByPersistence(Input=source)
simplified.InputArray = ["POINTS", "Scalars_"]
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1
simplified.PairType = 0  # all pair types
simplified.UpdatePipeline()

diagram = TTKPersistenceDiagram(Input=simplified)
diagram.ScalarField = ["POINTS", "Scalars_"]
diagram.UpdatePipeline()

# Retrieve the pair endpoints and pair dimension from TTK's diagram geometry.
data = servermanager.Fetch(diagram)
pd = data.GetPointData()
cd = data.GetCellData()
print("point arrays:", [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())])
print("cell arrays:", [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())])
print("points/cells:", data.GetNumberOfPoints(), data.GetNumberOfCells())

pair_type = vtk_to_numpy(cd.GetArray("PairType"))
pair_identifier = vtk_to_numpy(cd.GetArray("PairIdentifier"))
coords = vtk_to_numpy(data.GetPoints().GetData())

# Each persistence pair is represented by a two-point line in birth/death space.
birth, death, order = [], [], []
for cell_id in range(data.GetNumberOfCells()):
    cell = data.GetCell(cell_id)
    if cell.GetNumberOfPoints() != 2:
        continue
    dim = int(pair_type[cell_id])
    if dim not in (0, 2):
        continue
    a, b = cell.GetPointId(0), cell.GetPointId(1)
    # TTK embeds the diagram as (birth, death, 0).
    birth.append(coords[a, 0])
    death.append(coords[b, 1])
    order.append(dim)

birth, death, order = map(np.asarray, (birth, death, order))
finite = np.isfinite(birth) & np.isfinite(death)
birth, death, order = birth[finite], death[finite], order[finite]
print("rendered pairs by order:", {d: int(np.count_nonzero(order == d)) for d in (0, 2)})
print("value range:", float(np.min(np.r_[birth, death])), float(np.max(np.r_[birth, death])))

fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
lo, hi = float(np.min(np.r_[birth, death])), float(np.max(np.r_[birth, death]))
pad = max((hi - lo) * 0.06, 0.01)
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.4", lw=1.1, label="diagonal")
style = {0: ("#2563eb", "Order 0"), 2: ("#dc2626", "Order 2")}
for d in (0, 2):
    m = order == d
    ax.scatter(birth[m], death[m], s=26, alpha=0.85, color=style[d][0], label=style[d][1], edgecolors="none")
ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), aspect="equal",
       xlabel="Birth", ylabel="Death",
       title="Sublevel-set persistence diagram — simplification threshold 0.04")
ax.grid(True, color="0.9")
ax.legend(loc="upper left", frameon=True)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
