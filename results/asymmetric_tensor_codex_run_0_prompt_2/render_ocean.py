#!/opt/conda/bin/pvpython
"""Render the Ocean tensor eigenvector partition and its PL degeneracies."""
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray, vtkIntArray
from vtkmodules.vtkCommonDataModel import vtkCellArray
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter, vtkPolyDataMapper, vtkActor, vtkTexture
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource, vtkPlaneSource
from vtkmodules.vtkIOImage import vtkPNGWriter

INPUT = 'Ocean.vti'
PNG = 'Ocean_eigenvector_partition.png'
POINTS = 'Ocean_degenerate_points.vtp'
SAMPLES_PER_SQUARE = 10

r = vtkXMLImageDataReader()
r.SetFileName(INPUT)
r.Update()
data = r.GetOutput()
nx, ny, _ = data.GetDimensions()
x0, x1, y0, y1, _, _ = data.GetBounds()

def scalar(name):
    # VTK image point ids advance x first, matching reshape (y, x).
    return np.array(data.GetPointData().GetArray(name)).reshape(ny, nx)

A, B, C, D = (scalar(n) for n in ('A', 'B', 'C', 'D'))

# A line field is controlled by the symmetric traceless tensor (q, u).
# Its zeros are the isolated degeneracies of the eigenvector partition.
q = A - D
u = B + C

# Locate all roots exactly in the PL triangulation (two triangles per square).
roots = []
for j in range(ny - 1):
    for i in range(nx - 1):
        for corners in ((0, 1, 3), (0, 3, 2)):
            # Corner numbering: 0=(i,j), 1=(i+1,j), 2=(i,j+1), 3=(i+1,j+1)
            xy = np.array(((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)), float)[list(corners)]
            vv = np.array([(q[j, i], u[j, i]), (q[j, i + 1], u[j, i + 1]),
                           (q[j + 1, i], u[j + 1, i]), (q[j + 1, i + 1], u[j + 1, i + 1])])[list(corners)]
            M = np.column_stack((np.ones(3), xy))
            coef = np.linalg.solve(M, vv)  # rows: constant, dx, dy; columns q,u
            J = coef[1:, :].T
            try:
                p = np.linalg.solve(J, -coef[0])
            except np.linalg.LinAlgError:
                continue
            # Include a root on a shared edge only once.
            bary = np.linalg.solve(np.vstack((xy.T, np.ones(3))), np.r_[p, 1.0])
            if np.all(bary >= -1e-9) and np.all(bary <= 1 + 1e-9):
                if not any(np.linalg.norm(p - old[0]) < 1e-6 for old in roots):
                    # Positive half-index = wedge; negative half-index = trisector.
                    roots.append((p, 'wedge' if np.linalg.det(J) > 0 else 'trisector'))

# Rasterize at 10 pixels per original square. Values are bilinearly sampled from
# the VTI lattice, which preserves the supplied grid while giving a smooth display.
W = (nx - 1) * SAMPLES_PER_SQUARE
H = (ny - 1) * SAMPLES_PER_SQUARE
gx = (np.arange(W) + .5) / SAMPLES_PER_SQUARE
gy = (np.arange(H) + .5) / SAMPLES_PER_SQUARE
ix, iy = gx.astype(int), gy.astype(int)
fx, fy = gx - ix, gy - iy

def bilinear(f):
    f00, f10 = f[iy[:, None], ix[None, :]], f[iy[:, None], ix[None, :] + 1]
    f01, f11 = f[iy[:, None] + 1, ix[None, :]], f[iy[:, None] + 1, ix[None, :] + 1]
    return (1-fy)[:, None]*((1-fx)[None, :]*f00 + fx[None, :]*f10) + fy[:, None]*((1-fx)[None, :]*f01 + fx[None, :]*f11)

a, b, c, d = (bilinear(f) for f in (A, B, C, D))
# Principal eigenvector of a general 2x2 real matrix. Hue encodes unoriented angle.
disc = ((a-d)*.5)**2 + b*c
real = disc >= 0
lam = (a+d)*.5 + np.sqrt(np.maximum(disc, 0))
vx, vy = b, lam-a
use_alt = (vx*vx + vy*vy) < 1e-16
vx[use_alt], vy[use_alt] = lam[use_alt]-d[use_alt], c[use_alt]
theta = np.mod(np.arctan2(vy, vx), np.pi)
h = theta / np.pi
# Cyclic, high-contrast line-field coloring (black denotes no real eigendirection).
rgb = np.empty((H, W, 3), dtype=np.uint8)
for k, shift in enumerate((0, 2/3, 1/3)):
    rgb[..., k] = np.clip(255*(.18 + .82*(.5 + .5*np.cos(2*np.pi*(h + shift)))), 0, 255)
rgb[~real] = (20, 20, 25)

image = vtkImageData()
image.SetDimensions(W, H, 1)
image.SetOrigin(x0, y0, 0)
image.SetSpacing((x1-x0)/W, (y1-y0)/H, 1)
image.GetPointData().SetScalars(numpy_to_vtk(rgb.reshape(-1, 3), deep=True, array_type=vtkUnsignedCharArray().GetDataType()))

# Persist the classified points as a reusable ParaView data set.
pd = vtkPolyData(); pts = vtkPoints(); verts = vtkCellArray(); types = vtkIntArray(); types.SetName('Type')
for p, kind in roots:
    pid = pts.InsertNextPoint(float(p[0]), float(p[1]), 0.0)
    verts.InsertNextCell(1); verts.InsertCellPoint(pid)
    types.InsertNextValue(1 if kind == 'trisector' else 0)
pd.SetPoints(pts); pd.SetVerts(verts); pd.GetPointData().AddArray(types)
w = vtkXMLPolyDataWriter(); w.SetFileName(POINTS); w.SetInputData(pd); w.Write()

ren = vtkRenderer(); ren.SetBackground(.12, .12, .12)
# A textured plane preserves the RGB partition faithfully in off-screen rendering.
plane = vtkPlaneSource(); plane.SetOrigin(x0, y0, 0); plane.SetPoint1(x1, y0, 0); plane.SetPoint2(x0, y1, 0); plane.Update()
texture = vtkTexture(); texture.SetInputData(image); texture.InterpolateOff()
ia = vtkActor(); ia.SetMapper(vtkPolyDataMapper()); ia.GetMapper().SetInputConnection(plane.GetOutputPort()); ia.SetTexture(texture); ren.AddActor(ia)
for p, kind in roots:
    circle = vtkRegularPolygonSource(); circle.SetCenter(float(p[0]), float(p[1]), .15); circle.SetRadius(1.0); circle.SetNumberOfSides(48); circle.GeneratePolygonOn(); circle.Update()
    mapper = vtkPolyDataMapper(); mapper.SetInputConnection(circle.GetOutputPort())
    actor = vtkActor(); actor.SetMapper(mapper)
    if kind == 'trisector': actor.GetProperty().SetColor(1.0, .35, .65)  # pink
    else: actor.GetProperty().SetColor(1, 1, 1)  # white
    actor.GetProperty().SetEdgeVisibility(True); actor.GetProperty().SetEdgeColor(0, 0, 0); actor.GetProperty().SetLineWidth(1)
    ren.AddActor(actor)

win = vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetSize(W, H); win.AddRenderer(ren)
ren.ResetCamera(); cam = ren.GetActiveCamera(); cam.ParallelProjectionOn(); cam.SetParallelScale((y1-y0)/2); cam.SetPosition(50, 50, 100); cam.SetFocalPoint(50, 50, 0); ren.ResetCameraClippingRange()
win.Render()
capture = vtkWindowToImageFilter(); capture.SetInput(win); capture.SetInputBufferTypeToRGBA(); capture.ReadFrontBufferOff(); capture.Update()
writer = vtkPNGWriter(); writer.SetFileName(PNG); writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
print(f'Wrote {PNG} ({W}x{H}) and {POINTS}; {len(roots)} degeneracies: ' +
      f'{sum(k == "wedge" for _, k in roots)} wedges, {sum(k == "trisector" for _, k in roots)} trisectors')
