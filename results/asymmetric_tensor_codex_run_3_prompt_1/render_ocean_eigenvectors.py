"""Render the eigenvector partition of Ocean.vti with classified degeneracies."""
import colorsys
import math
import numpy as np

from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLImageDataWriter
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersSources import vtkRegularPolygonSource
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter, vtkPolyDataMapper, vtkActor, vtkImageActor
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

INPUT = "Ocean.vti"
PNG = "Ocean_eigenvector_partition.png"
DATA = "Ocean_eigenvector_partition.vti"

reader = vtkXMLImageDataReader()
reader.SetFileName(INPUT)
reader.Update()
source = reader.GetOutput()
pd = source.GetPointData()
nx, ny, _ = source.GetDimensions()
origin = source.GetOrigin()
spacing = source.GetSpacing()

def values(name):
    # VTK image point order is x-fastest.
    return vtk_to_numpy(pd.GetArray(name)).reshape(ny, nx)

A, B, C, D = (values(n) for n in ("A", "B", "C", "D"))

# Ten pixel centres in every original grid square: exact requested resolution.
sub = 10
px = (np.arange((nx - 1) * sub) + .5) / sub
py = (np.arange((ny - 1) * sub) + .5) / sub
X, Y = np.meshgrid(px, py)
ix, iy = X.astype(int), Y.astype(int)
u, v = X - ix, Y - iy

def bilinear(field):
    return ((1-u)*(1-v)*field[iy, ix] + u*(1-v)*field[iy, ix+1]
            + (1-u)*v*field[iy+1, ix] + u*v*field[iy+1, ix+1])

a, b, c, d = (bilinear(q) for q in (A, B, C, D))
disc = (a-d)**2 + 4*b*c
real = disc >= 0
root = np.sqrt(np.maximum(disc, 0))
lam = .5 * (a + d + root)

# Stable right-eigenvector formula for the larger real eigenvalue.
vx, vy = b.copy(), lam-a
altx, alty = lam-d, c
use_alt = vx*vx + vy*vy < altx*altx + alty*alty
vx[use_alt], vy[use_alt] = altx[use_alt], alty[use_alt]
angle = np.mod(np.arctan2(vy, vx), math.pi) / math.pi

# Orientation colour wheel; complex-eigenvalue regions are intentionally muted.
rgb = np.empty((len(py), len(px), 3), dtype=np.uint8)
rgb[:] = (21, 28, 39)
for j, i in zip(*np.where(real)):
    rgb[j, i] = np.array(colorsys.hsv_to_rgb(float(angle[j, i]), .82, .94)) * 255

# Store the 10 px/square partition as a reusable VTK image.
image = vtkImageData()
image.SetDimensions(len(px), len(py), 1)
image.SetOrigin(origin[0] + .05, origin[1] + .05, 0)
image.SetSpacing(spacing[0] / sub, spacing[1] / sub, 1)
arr = numpy_to_vtk(np.ascontiguousarray(rgb.reshape(-1, 3)), deep=True, array_type=3)
arr.SetName("EigenvectorPartitionRGB")
arr.SetNumberOfComponents(3)
image.GetPointData().SetScalars(arr)
writer = vtkXMLImageDataWriter()
writer.SetFileName(DATA)
writer.SetInputData(image)
writer.Write()

# For line-field singularities use F=A-D and G=B+C.  The zero contours of F,G
# intersect at isolated degeneracies; the Jacobian sign gives the half-index:
# + is a wedge and - is a trisector.
F, G = A-D, B+C
trisectors, wedges = [], []
for y in range(ny-1):
    for x in range(nx-1):
        # Split every square along its lower-left to upper-right diagonal.
        for ids in ((0, 1, 3), (0, 3, 2)):
            coords = np.array([(0.,0.), (1.,0.), (0.,1.), (1.,1.)])[list(ids)]
            fv = np.array([F[y+int(q[1]), x+int(q[0])] for q in coords])
            gv = np.array([G[y+int(q[1]), x+int(q[0])] for q in coords])
            M = np.column_stack((np.ones(3), coords))
            try:
                cf = np.linalg.solve(M, fv); cg = np.linalg.solve(M, gv)
                q = np.linalg.solve(np.array([[cf[1], cf[2]], [cg[1], cg[2]]]), -np.array([cf[0], cg[0]]))
            except np.linalg.LinAlgError:
                continue
            # Half-open containment prevents counting points on the diagonal twice.
            bary = np.linalg.solve(np.vstack((coords.T, np.ones(3))), np.r_[q, 1])
            if np.all(bary >= -1e-8) and np.all(bary < 1-1e-8):
                p = (origin[0] + spacing[0]*(x+q[0]), origin[1] + spacing[1]*(y+q[1]))
                (wedges if np.linalg.det(np.array([[cf[1], cf[2]], [cg[1], cg[2]]])) > 0 else trisectors).append(p)

def disks(points, color):
    # Radius is in input-data units, exactly 1 as requested.
    actors = []
    for x, y in points:
        disk = vtkRegularPolygonSource()
        disk.SetCenter(x, y, .2)
        disk.SetRadius(1.0)
        disk.SetNumberOfSides(48)
        disk.Update()
        mapper = vtkPolyDataMapper(); mapper.SetInputConnection(disk.GetOutputPort())
        actor = vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetEdgeColor(0.08, 0.08, 0.08)
        actor.GetProperty().SetLineWidth(1.0)
        actors.append(actor)
    return actors

renderer = vtkRenderer()
renderer.SetBackground(0.06, .075, .10)
actor = vtkImageActor(); actor.SetInputData(image); renderer.AddActor(actor)
for overlay in disks(trisectors, (1.0, .38, .66)) + disks(wedges, (1., 1., 1.)):
    renderer.AddActor(overlay)

window = vtkRenderWindow()
window.SetOffScreenRendering(1)
window.SetSize((nx-1)*sub, (ny-1)*sub)
window.AddRenderer(renderer)
camera = renderer.GetActiveCamera()
camera.SetParallelProjection(True)
camera.SetFocalPoint(50, 50, 0)
camera.SetPosition(50, 50, 100)
camera.SetViewUp(0, 1, 0)
# Off-screen ParaView uses a two-pixel backing scale in this environment;
# a scale of 50 yields the requested 1000-pixel-square data viewport.
camera.SetParallelScale(50)
renderer.ResetCameraClippingRange()
window.Render()
capture = vtkWindowToImageFilter(); capture.SetInput(window); capture.SetInputBufferTypeToRGBA(); capture.ReadFrontBufferOff(); capture.Update()
from vtkmodules.vtkIOImage import vtkPNGWriter
png = vtkPNGWriter(); png.SetFileName(PNG); png.SetInputConnection(capture.GetOutputPort()); png.Write()
print(f"Rendered {PNG} at {window.GetSize()[0]}x{window.GetSize()[1]} pixels")
print(f"Degenerate points: {len(trisectors)} trisectors, {len(wedges)} wedges")
