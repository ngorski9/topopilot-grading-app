"""Render Ocean.vti's 2-D eigenvector line-field partition.

The input stores the in-plane tensor entries A, B, C and D.  The unoriented
principal eigendirection is represented by theta = 1/2 atan2(B+C, A-D).
Its zeros are located exactly in each piecewise-linear triangle and classified
from the sign of the local Jacobian: positive = wedge, negative = trisector.
"""
import colorsys
import math
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from PIL import Image, ImageDraw

INPUT = "Ocean.vti"
OUTPUT = "Ocean_eigenvector_partition.png"
SAMPLES_PER_CELL = 10


def field_arrays():
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(INPUT)
    reader.Update()
    image = reader.GetOutput()
    nx, ny, _ = image.GetDimensions()
    point_data = image.GetPointData()
    # VTK's point order is x-fastest; the image array is indexed [y, x].
    arrays = [vtk_to_numpy(point_data.GetArray(name)).reshape(ny, nx)
              for name in ("A", "B", "C", "D")]
    return arrays, nx, ny


def roots_and_types(p, q):
    """Locate P=Q=0 in the two triangles of every grid square."""
    roots = []
    ny, nx = p.shape
    for y in range(ny - 1):
        for x in range(nx - 1):
            values = [(x, y, p[y, x], q[y, x]),
                      (x + 1, y, p[y, x + 1], q[y, x + 1]),
                      (x + 1, y + 1, p[y + 1, x + 1], q[y + 1, x + 1]),
                      (x, y + 1, p[y + 1, x], q[y + 1, x])]
            # A consistent SW-to-NE split makes the supplied PL field explicit.
            for tri in ((0, 1, 2), (0, 2, 3)):
                xyz = np.array([[values[i][0], values[i][1], 1.0] for i in tri])
                pv = np.array([values[i][2] for i in tri])
                qv = np.array([values[i][3] for i in tri])
                try:
                    cp = np.linalg.solve(xyz, pv)
                    cq = np.linalg.solve(xyz, qv)
                    weights = np.linalg.solve(np.vstack((pv, qv, np.ones(3))),
                                              np.array([0.0, 0.0, 1.0]))
                except np.linalg.LinAlgError:
                    continue
                if np.all(weights >= -1.e-9):
                    px = sum(weights[k] * values[i][0] for k, i in enumerate(tri))
                    py = sum(weights[k] * values[i][1] for k, i in enumerate(tri))
                    jacobian = cp[0] * cq[1] - cp[1] * cq[0]
                    roots.append((px, py, "wedge" if jacobian > 0 else "trisector"))
    return roots


def main():
    (a, b, c, d), nx, ny = field_arrays()
    p, q = a - d, b + c
    width, height = (nx - 1) * SAMPLES_PER_CELL, (ny - 1) * SAMPLES_PER_CELL

    # Piecewise-linear interpolation on the same SW-to-NE triangle split.
    yy, xx = np.indices((height, width), dtype=float)
    gx, gy = (xx + .5) / SAMPLES_PER_CELL, (yy + .5) / SAMPLES_PER_CELL
    ix, iy = np.floor(gx).astype(int), np.floor(gy).astype(int)
    fx, fy = gx - ix, gy - iy
    p00, p10, p01, p11 = p[iy, ix], p[iy, ix + 1], p[iy + 1, ix], p[iy + 1, ix + 1]
    q00, q10, q01, q11 = q[iy, ix], q[iy, ix + 1], q[iy + 1, ix], q[iy + 1, ix + 1]
    lower = fy <= fx
    pp = np.where(lower, p00 + fx * (p10 - p00) + fy * (p11 - p10),
                  p00 + fx * (p11 - p01) + fy * (p01 - p00))
    qq = np.where(lower, q00 + fx * (q10 - q00) + fy * (q11 - q10),
                  q00 + fx * (q11 - q01) + fy * (q01 - q00))
    theta = .5 * np.arctan2(qq, pp)

    # A cyclic hue makes the pi-periodic (unoriented) eigenvector partition clear.
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    hue = (theta + math.pi / 2.) / math.pi
    for j in range(height):
        for i in range(width):
            rgb[j, i] = [round(255 * z) for z in colorsys.hsv_to_rgb(hue[j, i], .72, .92)]
    # Convert NumPy's y=0 row (the data's lower edge) to display coordinates.
    canvas = Image.fromarray(np.flipud(rgb), "RGB")
    draw = ImageDraw.Draw(canvas)
    roots = roots_and_types(p, q)
    for x, y, kind in roots:
        cx, cy = x * SAMPLES_PER_CELL, height - y * SAMPLES_PER_CELL
        radius = SAMPLES_PER_CELL  # radius 1 in field coordinates
        colour = (255, 105, 180) if kind == "trisector" else (255, 255, 255)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                     fill=colour, outline=(35, 35, 35), width=1)
    canvas.save(OUTPUT)
    wedges = sum(kind == "wedge" for _, _, kind in roots)
    print(f"Wrote {OUTPUT}: {width}x{height}; {len(roots)} degeneracies "
          f"({wedges} wedges, {len(roots)-wedges} trisectors).")


if __name__ == "__main__":
    main()
