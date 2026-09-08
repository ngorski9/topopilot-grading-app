import vtk
import numpy as np
from vtk.util import numpy_support

input_file = '/workspace/brain.vti'
output_file = '/workspace/brain_degenerate_points.png'

# Read arrays A, B, D of the symmetric tensor [[A,B],[B,D]].
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(input_file)
reader.Update()
image = reader.GetOutput()
pd = image.GetPointData()
extent = image.GetExtent()
nx, ny = extent[1] - extent[0] + 1, extent[3] - extent[2] + 1
origin, spacing = image.GetOrigin(), image.GetSpacing()
A = numpy_support.vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
B = numpy_support.vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
D = numpy_support.vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

# Degeneracies are the simultaneous zeros of f=(A-D, 2B).  On each of the
# two PL triangles in a pixel, solve f(p)=0 using barycentric coordinates.
f0, f1 = A-D, 2*B
points, kinds = [], []
for j in range(ny-1):
    for i in range(nx-1):
        # Counter-clockwise triangles; avoiding duplicate vertices naturally
        # follows from retaining only a single valid solution per triangle.
        for tri in ((0, 1, 3), (0, 3, 2)):
            ids = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
            v = [ids[k] for k in tri]
            vals = np.array([[f0[y,x], f1[y,x]] for x,y in v], dtype=float)
            M = np.column_stack((vals[1]-vals[0], vals[2]-vals[0]))
            det = np.linalg.det(M)
            if abs(det) < 1e-12:
                continue
            uv = np.linalg.solve(M, -vals[0])
            bary = np.array([1-uv[0]-uv[1], uv[0], uv[1]])
            if np.min(bary) >= -1e-9 and np.max(bary) <= 1+1e-9:
                # Map barycentric location to physical coordinates.
                xy = bary @ np.array(v, dtype=float)
                x, y = origin[0] + spacing[0]*xy[0], origin[1] + spacing[1]*xy[1]
                # For a tensor line field, positive/negative index correspond
                # respectively to wedges/trisectors. det(M) differs from the
                # physical Jacobian only by a positive triangle-area factor.
                points.append((x, y, origin[2]))
                kinds.append('wedge' if det > 0 else 'trisector')

# Remove rare double detections at shared mesh vertices.
unique = []
for p, k in zip(points, kinds):
    if not any(np.hypot(p[0]-q[0], p[1]-q[1]) < 1e-6 for q, _ in unique):
        unique.append((p, k))
points, kinds = [u[0] for u in unique], [u[1] for u in unique]

# Background: tensor anisotropy sqrt((A-D)^2+(2B)^2), which vanishes exactly
# at degeneracies and reveals the tensor structure across the slice.
anis = np.sqrt(f0*f0 + f1*f1)
bg = vtk.vtkImageData(); bg.DeepCopy(image)
arr = numpy_support.numpy_to_vtk(anis.ravel(), deep=True)
arr.SetName('Tensor anisotropy')
bg.GetPointData().AddArray(arr); bg.GetPointData().SetScalars(arr)

mapper = vtk.vtkDataSetMapper(); mapper.SetInputData(bg); mapper.SetScalarModeToUsePointFieldData(); mapper.SelectColorArray('Tensor anisotropy')
lo, hi = np.percentile(anis, [2, 98]); mapper.SetScalarRange(lo, hi)
lut = vtk.vtkLookupTable(); lut.SetNumberOfTableValues(256); lut.SetRange(lo, hi); lut.Build()
for n in range(256):
    t=n/255; lut.SetTableValue(n, 0.035+0.45*t, 0.045+0.15*t, 0.11+0.42*t, 1)
mapper.SetLookupTable(lut)
field_actor = vtk.vtkActor(); field_actor.SetMapper(mapper)

def sphere_actor(selected, color):
    poly = vtk.vtkPolyData(); pts = vtk.vtkPoints()
    for idx in selected: pts.InsertNextPoint(points[idx])
    poly.SetPoints(pts)
    sphere = vtk.vtkSphereSource(); sphere.SetRadius(1.0); sphere.SetThetaResolution(20); sphere.SetPhiResolution(14)
    glyph = vtk.vtkGlyph3D(); glyph.SetInputData(poly); glyph.SetSourceConnection(sphere.GetOutputPort()); glyph.ScalingOff(); glyph.Update()
    m = vtk.vtkPolyDataMapper(); m.SetInputConnection(glyph.GetOutputPort())
    a = vtk.vtkActor(); a.SetMapper(m); a.GetProperty().SetColor(color); a.GetProperty().SetSpecular(.35); a.GetProperty().SetSpecularPower(20)
    return a

wedges = [i for i,k in enumerate(kinds) if k == 'wedge']
trisectors = [i for i,k in enumerate(kinds) if k == 'trisector']

ren = vtk.vtkRenderer(); ren.SetBackground(0.008, 0.01, 0.025); ren.AddActor(field_actor)
ren.AddActor(sphere_actor(wedges, (1,1,1)))
ren.AddActor(sphere_actor(trisectors, (1.0, 0.2, 0.62)))
window = vtk.vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1500, 900); window.AddRenderer(ren)
ren.ResetCamera(); cam=ren.GetActiveCamera(); cam.ParallelProjectionOn(); cam.SetParallelScale(ny*0.58); cam.SetPosition(nx/2, ny/2, 180); cam.SetFocalPoint(nx/2, ny/2, 0); cam.SetViewUp(0,1,0)
ren.ResetCameraClippingRange(); window.Render()

# A compact in-scene legend.
for text, pos, color in [('white: wedge', (18, 12), (1,1,1)), ('pink: trisector', (18, 35), (1,.2,.62))]:
    actor=vtk.vtkTextActor(); actor.SetInput(text); actor.GetTextProperty().SetFontSize(20); actor.GetTextProperty().SetColor(color); actor.SetDisplayPosition(*pos); ren.AddActor2D(actor)
window.Render()
w = vtk.vtkWindowToImageFilter(); w.SetInput(window); w.ReadFrontBufferOff(); w.Update()
writer=vtk.vtkPNGWriter(); writer.SetFileName(output_file); writer.SetInputConnection(w.GetOutputPort()); writer.Write()
print(f'Detected {len(points)} degeneracies: {len(wedges)} wedges, {len(trisectors)} trisectors')
print(output_file)
