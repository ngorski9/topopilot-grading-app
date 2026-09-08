"""Render the real-eigenvector partition of Ocean.vti at 10 px/grid square."""
from pathlib import Path
import colorsys
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLImageDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Ocean.vti"
OUT = ROOT / "Ocean_eigenvector_partition.png"
PTS = ROOT / "Ocean_degenerate_points.csv"
SCALE = 10


def load_components():
    reader = vtkXMLImageDataReader()
    reader.SetFileName(str(INPUT))
    reader.Update()
    data = reader.GetOutput()
    nx, ny, _ = data.GetDimensions()
    pd = data.GetPointData()
    # VTK point order is x-fastest; image arrays use [y, x].
    return [vtk_to_numpy(pd.GetArray(name)).reshape(ny, nx) for name in "ABCD"]


def bilinear(values, size):
    """Evaluate a vertex field on a regular output lattice."""
    ny, nx = values.shape
    x = np.arange((nx - 1) * size, dtype=float) / size
    y = np.arange((ny - 1) * size, dtype=float) / size
    ix, iy = x.astype(int), y.astype(int)
    fx, fy = x - ix, y - iy
    v00 = values[iy[:, None], ix]
    v10 = values[iy[:, None], ix + 1]
    v01 = values[iy[:, None] + 1, ix]
    v11 = values[iy[:, None] + 1, ix + 1]
    return ((1 - fy[:, None]) * ((1 - fx) * v00 + fx * v10)
            + fy[:, None] * ((1 - fx) * v01 + fx * v11))


def classify_point(A, B, C, D, x, y):
    """Classify by the local line-field index (positive=wedge)."""
    ix, iy = min(int(x), 99), min(int(y), 99)
    fx, fy = x - ix, y - iy
    # Symmetric traceless part controls the unoriented eigenline index.
    p = A - D
    q = B + C
    def gradients(v):
        vx = (1 - fy) * (v[iy, ix + 1] - v[iy, ix]) + fy * (v[iy + 1, ix + 1] - v[iy + 1, ix])
        vy = (1 - fx) * (v[iy + 1, ix] - v[iy, ix]) + fx * (v[iy + 1, ix + 1] - v[iy, ix + 1])
        return vx, vy
    px, py = gradients(p)
    qx, qy = gradients(q)
    return "wedge" if px * qy - py * qx >= 0 else "trisector"


def main():
    A, B, C, D = load_components()
    # Degeneracies are the exact repeated-eigenvalue samples, Dscr=(A-D)^2+4BC.
    discr_nodes = (A - D) ** 2 + 4 * B * C
    degeneracies = []
    for y, x in np.argwhere(np.isclose(discr_nodes, 0.0, atol=1e-14)):
        degeneracies.append((float(x), float(y), classify_point(A, B, C, D, x, y)))

    a, b, c, d = [bilinear(v, SCALE) for v in (A, B, C, D)]
    delta = (a - d) ** 2 + 4 * b * c
    real = delta > 1e-12
    root = np.zeros_like(delta)
    root[real] = np.sqrt(delta[real])
    # A stable right eigenvector for the larger real eigenvalue.
    vx, vy = b.copy(), (d - a + root) / 2
    weak = np.hypot(vx, vy) < 1e-10
    vx[weak], vy[weak] = (a - d + root)[weak] / 2, c[weak]
    angle = np.mod(np.arctan2(vy, vx), np.pi) / np.pi

    # Cyclic line-direction colour map: equal colours indicate equal eigenlines.
    hsv = np.stack((angle, np.full_like(angle, .68), np.full_like(angle, .93)), axis=-1)
    rgb = np.empty((*angle.shape, 3), dtype=np.uint8)
    for row in range(rgb.shape[0]):
        rgb[row] = (np.array([colorsys.hsv_to_rgb(*h) for h in hsv[row]]) * 255).astype(np.uint8)
    # Complex-eigenvalue region has no real eigenvector partition.
    rgb[~real] = (24, 35, 52)
    image = Image.fromarray(rgb[::-1], "RGB")
    draw = ImageDraw.Draw(image)
    radius = SCALE  # one grid-square radius at 10 pixels per square
    for x, y, kind in degeneracies:
        cx, cy = int(round(x * SCALE)), int(round((100 - y) * SCALE))
        colour = (255, 105, 180) if kind == "trisector" else (255, 255, 255)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour, outline=(30, 30, 30), width=1)
    image.save(OUT)
    PTS.write_text("x,y,type\n" + "\n".join(f"{x:g},{y:g},{kind}" for x, y, kind in degeneracies) + "\n")
    print(f"wrote {OUT} ({image.width}x{image.height}); {len(degeneracies)} degeneracies")


if __name__ == "__main__":
    main()
