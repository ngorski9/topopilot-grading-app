"""Render the H0/H2 persistence diagram of QMCPACK.vti with cutoff 0.04."""
from paraview.simple import XMLImageDataReader, TTKPersistenceDiagram, UpdatePipeline
from paraview import servermanager
import csv
import matplotlib.pyplot as plt

INPUT = "/workspace/QMCPACK.vti"
OUT_PNG = "/workspace/QMCPACK_persistence_diagram_H0_H2_cutoff_0.04.png"
OUT_CSV = "/workspace/QMCPACK_persistence_pairs_H0_H2_cutoff_0.04.csv"
CUTOFF = 0.04

# TTK's persistence-diagram filter computes the PL sub-level-set filtration on
# the ImageData grid.  A persistence simplification at CUTOFF is effected by
# retaining only pairs whose persistence is at least CUTOFF.
reader = XMLImageDataReader(FileName=[INPUT])
diagram = TTKPersistenceDiagram(Input=reader)
diagram.ScalarField = ["POINTS", "Scalars_"]
diagram.Dimensions = "All Dimensions"
UpdatePipeline(proxy=diagram)
grid = servermanager.Fetch(diagram)
cell_data, points = grid.GetCellData(), grid.GetPoints()

pairs = []
for i in range(grid.GetNumberOfCells()):
    dimension = int(cell_data.GetArray("PairType").GetValue(i))
    persistence = float(cell_data.GetArray("Persistence").GetValue(i))
    if dimension not in (0, 2) or persistence < CUTOFF:
        continue
    birth = float(cell_data.GetArray("Birth").GetValue(i))
    cell = grid.GetCell(i)
    death = float(points.GetPoint(cell.GetPointIds().GetId(1))[1])
    pairs.append({
        "dimension": dimension,
        "birth": birth,
        "death": death,
        "persistence": persistence,
        "is_finite": int(cell_data.GetArray("IsFinite").GetValue(i)),
    })

with open(OUT_CSV, "w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["dimension", "birth", "death", "persistence", "is_finite"])
    writer.writeheader()
    writer.writerows(pairs)

fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=180)
ax.set_facecolor("#fbfbfd")
limit = max(max(p["death"] for p in pairs), max(p["birth"] for p in pairs))
pad = 0.05 * (limit if limit else 1.0)
ax.plot([-pad, limit + pad], [-pad, limit + pad], color="#6b7280", lw=1.25, ls="--", label="Diagonal (birth = death)")

styles = {
    0: ("#2563eb", r"$H_0$ (components)"),
    2: ("#dc2626", r"$H_2$ (voids)"),
}
for dimension, (color, label) in styles.items():
    subset = [p for p in pairs if p["dimension"] == dimension]
    ax.scatter([p["birth"] for p in subset], [p["death"] for p in subset],
               s=58, c=color, edgecolors="white", linewidths=0.7, alpha=0.95,
               label=f"{label}  (n={len(subset)})", zorder=3)

ax.set_xlim(-pad, limit + pad)
ax.set_ylim(-pad, limit + pad)
ax.set_aspect("equal", adjustable="box")
ax.grid(True, color="#d1d5db", lw=0.65, alpha=0.65)
ax.set_xlabel("Birth scalar value")
ax.set_ylabel("Death scalar value")
ax.set_title("QMCPACK — piecewise-linear sub-level-set persistence diagram\n"
             "Persistence simplification cutoff = 0.04; displayed homology: $H_0$, $H_2$")
ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT_PNG, bbox_inches="tight")
print(f"saved {OUT_PNG}")
print(f"saved {OUT_CSV}")
print("counts: " + ", ".join(f"H{d}={sum(p['dimension'] == d for p in pairs)}" for d in (0, 2)))
