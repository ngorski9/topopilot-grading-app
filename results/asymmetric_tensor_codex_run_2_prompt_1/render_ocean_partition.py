import numpy as np
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataWriter
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray, vtkLookupTable
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonDataModel import VTK_LINE
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersGeometry import vtkImageDataGeometryFilter
from vtkmodules.vtkRenderingCore import vtkRenderer, vtkRenderWindow, vtkWindowToImageFilter
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkDataSetMapper, vtkActor
from vtkmodules.vtkRenderingAnnotation import vtkScalarBarActor
from vtkmodules.vtkRenderingCore import vtkTextProperty
from vtkmodules.vtkIOImage import vtkPNGWriter

INFILE = '/workspace/Ocean.vti'
OUTPNG = '/workspace/Ocean_eigenvector_partition.png'

r = vtkXMLImageDataReader(); r.SetFileName(INFILE); r.Update()
data = r.GetOutput(); nx, ny, _ = data.GetDimensions(); pd = data.GetPointData()
a = vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
b = vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
c = vtk_to_numpy(pd.GetArray('C')).reshape(ny, nx)
d = vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

# Zhang--Pang decomposition: isotropic + rotation + symmetric anisotropy.
gr = (c - b) * .5
gx = (a - d) * .5
gy = (b + c) * .5
gs = np.sqrt(gx * gx + gy * gy)

# Eigenvector partition: real/stretching where anisotropy dominates; otherwise
# the two rotational hemispheres (counter-clockwise and clockwise).
N = 1001  # 10 pixels per original unit square, plus end point
u = np.linspace(0, 100, N); xx, yy = np.meshgrid(u, u)
ix = np.minimum(xx.astype(int), nx - 2); iy = np.minimum(yy.astype(int), ny - 2)
tx, ty = xx - ix, yy - iy
def bilin(q):
    return ((1-tx)*(1-ty)*q[iy,ix] + tx*(1-ty)*q[iy,ix+1]
            + (1-tx)*ty*q[iy+1,ix] + tx*ty*q[iy+1,ix+1])
rg, sgx, sgy = bilin(gr), bilin(gx), bilin(gy)
sg = np.sqrt(sgx*sgx + sgy*sgy)
part = np.where(sg >= np.abs(rg), 0, np.where(rg >= 0, 1, 2)).astype(np.uint8)

# RGB is used directly to retain the categorical partition colors.
colors = np.array([[46, 125, 150], [246, 173, 67], [116, 86, 156]], dtype=np.uint8)
rgb = colors[part].reshape(-1, 3)
img = vtkImageData(); img.SetDimensions(N, N, 1); img.SetSpacing(.1, .1, 1)
arr = numpy_to_vtk(rgb, deep=True, array_type=3); arr.SetNumberOfComponents(3); arr.SetName('Eigenvector partition')
img.GetPointData().SetScalars(arr)

# A PL cell is split into two triangles.  Solve gx=gy=0 in each triangle;
# det(grad(gx), grad(gy)) distinguishes wedge (+) from trisector (-).
pts = vtkPoints(); types = vtkUnsignedCharArray(); types.SetName('Type')
for j in range(ny-1):
    for i in range(nx-1):
        for tri in ((0, 1, 3), (0, 3, 2)):
            xy = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
            p0, p1, p2 = [xy[k] for k in tri]
            f = np.array([gx[y,x] for x,y in (p0,p1,p2)])
            g = np.array([gy[y,x] for x,y in (p0,p1,p2)])
            M = np.array([[p1[0]-p0[0], p1[1]-p0[1]], [p2[0]-p0[0], p2[1]-p0[1]]], float)
            try:
                cf = np.linalg.solve(M, np.array([f[1]-f[0], f[2]-f[0]]))
                cg = np.linalg.solve(M, np.array([g[1]-g[0], g[2]-g[0]]))
                G = np.array([[cf[0], cf[1]], [cg[0], cg[1]]])
                q = np.array(p0, float) + np.linalg.solve(G, -np.array([f[0], g[0]]))
            except np.linalg.LinAlgError:
                continue
            # Barycentric containment, allowing a tiny tolerance for shared edges.
            rel = q - np.array(p0); bc = np.linalg.solve(M.T, rel)
            if bc[0] >= -1e-8 and bc[1] >= -1e-8 and bc.sum() <= 1+1e-8:
                # avoid duplicate roots on the shared diagonal / grid edges
                candidate = np.array([q[0], q[1]])
                duplicate = any(np.linalg.norm(candidate - np.array(pts.GetPoint(k)[:2])) < 1e-5 for k in range(pts.GetNumberOfPoints()))
                if not duplicate:
                    pts.InsertNextPoint(float(q[0]), float(q[1]), .4)
                    types.InsertNextValue(0 if np.linalg.det(np.vstack((cf,cg))) < 0 else 1)

deg = vtkPolyData(); deg.SetPoints(pts); deg.GetPointData().AddArray(types); deg.GetPointData().SetScalars(types)
writer = vtkXMLPolyDataWriter(); writer.SetFileName('/workspace/Ocean_degenerate_points.vtp'); writer.SetInputData(deg); writer.Write()

# Make radius-1 circular markers in the field plane.
sphere = vtkSphereSource(); sphere.SetRadius(1.0); sphere.SetThetaResolution(24); sphere.SetPhiResolution(12)
glyph = vtkGlyph3D(); glyph.SetInputData(deg); glyph.SetSourceConnection(sphere.GetOutputPort()); glyph.ScalingOff(); glyph.Update()

lut = vtkLookupTable(); lut.SetNumberOfTableValues(2); lut.Build()
lut.SetTableValue(0, 1.0, 0.35, 0.65, 1.0)  # trisectors: pink
lut.SetTableValue(1, 1.0, 1.0, 1.0, 1.0)    # wedges: white
def marker_actor(kind, color):
    selected = vtkPolyData(); selected_pts = vtkPoints()
    for n in range(pts.GetNumberOfPoints()):
        if types.GetValue(n) == kind: selected_pts.InsertNextPoint(pts.GetPoint(n))
    selected.SetPoints(selected_pts)
    g = vtkGlyph3D(); g.SetInputData(selected); g.SetSourceConnection(sphere.GetOutputPort()); g.ScalingOff(); g.Update()
    m = vtkPolyDataMapper(); m.SetInputConnection(g.GetOutputPort()); m.ScalarVisibilityOff()
    a = vtkActor(); a.SetMapper(m); a.GetProperty().LightingOff(); a.GetProperty().SetColor(*color); return a
trisectors = marker_actor(0, (1.0, .35, .65)); wedges = marker_actor(1, (1.0, 1.0, 1.0))

surface = vtkImageDataGeometryFilter(); surface.SetInputData(img); surface.Update()
imapper = vtkDataSetMapper(); imapper.SetInputConnection(surface.GetOutputPort()); imapper.SetColorModeToDirectScalars(); imapper.ScalarVisibilityOn()
partition_actor = vtkActor(); partition_actor.SetMapper(imapper); partition_actor.GetProperty().LightingOff()
renderer = vtkRenderer(); renderer.SetBackground(0.08, 0.09, 0.12); renderer.AddActor(partition_actor); renderer.AddActor(trisectors); renderer.AddActor(wedges)
renderer.GetActiveCamera().SetPosition(50, 50, 180); renderer.GetActiveCamera().SetFocalPoint(50, 50, 0); renderer.GetActiveCamera().SetViewUp(0,1,0); renderer.GetActiveCamera().ParallelProjectionOn(); renderer.GetActiveCamera().SetParallelScale(110)

bar = vtkScalarBarActor(); bar.SetLookupTable(lut); bar.SetTitle('Degenerate points'); bar.SetNumberOfLabels(2); bar.SetPosition(.83,.13); bar.SetWidth(.13); bar.SetHeight(.26)
bar.GetTitleTextProperty().SetColor(1,1,1); bar.GetLabelTextProperty().SetColor(1,1,1); renderer.AddActor2D(bar)

window = vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1000,1000); window.AddRenderer(renderer); window.Render()
capture = vtkWindowToImageFilter(); capture.SetInput(window); capture.SetInputBufferTypeToRGBA(); capture.ReadFrontBufferOff(); capture.Update()
png = vtkPNGWriter(); png.SetFileName(OUTPNG); png.SetInputConnection(capture.GetOutputPort()); png.Write()
print('degenerate points:', pts.GetNumberOfPoints(), 'trisectors:', sum(types.GetValue(i)==0 for i in range(types.GetNumberOfTuples())), 'wedges:', sum(types.GetValue(i)==1 for i in range(types.GetNumberOfTuples())))
print(OUTPNG)
