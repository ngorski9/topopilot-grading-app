import csv
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

threshold = 0.04
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName('qmc_persistence_raw_port_0.vtu')
reader.Update()
grid = reader.GetOutput()
cell_data = grid.GetCellData()
pair_type = vtk_to_numpy(cell_data.GetArray('PairType'))
persistence = vtk_to_numpy(cell_data.GetArray('Persistence'))
birth = vtk_to_numpy(cell_data.GetArray('Birth'))
finite = vtk_to_numpy(cell_data.GetArray('IsFinite'))

# TTK PairType maps to homology dimensions: 0=min-saddle, 1=saddle-saddle, 2=saddle-max.
rows = []
for i, (dim, p, b, isfinite) in enumerate(zip(pair_type, persistence, birth, finite)):
    if int(dim) in (0, 2) and float(p) >= threshold:
        rows.append((i, int(dim), float(b), float(b + p), float(p), int(isfinite)))

with open('QMCPACK_persistence_simplified_0.04_H0_H2.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pair_id', 'homology_dimension', 'birth', 'death', 'persistence', 'is_finite'])
    w.writerows(rows)

colors = {0: '#1f77b4', 2: '#d62728'}
labels = {0: 'H0', 2: 'H2'}
fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
all_vals = [v for row in rows for v in row[2:4]]
lo, hi = min(all_vals), max(all_vals)
pad = max((hi - lo) * .06, .02)
ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color='#4b5563', lw=1.25, ls='--', label='birth = death')
for dim in (0, 2):
    pts = [r for r in rows if r[1] == dim]
    ax.scatter([r[2] for r in pts], [r[3] for r in pts], s=38, c=colors[dim],
               edgecolors='white', linewidths=.45, alpha=.9, label=f'{labels[dim]} (n={len(pts)})')
ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(lo-pad, hi+pad)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, color='#d1d5db', lw=.55, alpha=.7)
ax.set_xlabel('Birth'); ax.set_ylabel('Death')
ax.set_title('QMCPACK — Piecewise-Linear Sub-level Set Persistence Diagram\nPersistence simplification threshold = 0.04')
ax.legend(loc='upper left', frameon=True)
ax.text(.99, .02, 'TTK Discrete Morse Sandwich; dimensions 0 and 2 only', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=8, color='#374151')
fig.tight_layout()
fig.savefig('QMCPACK_persistence_diagram_simplified_0.04_H0_H2.png', facecolor='white')
print(f'kept={len(rows)} H0={sum(r[1] == 0 for r in rows)} H2={sum(r[1] == 2 for r in rows)}')
