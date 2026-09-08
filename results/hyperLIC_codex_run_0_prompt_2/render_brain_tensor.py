import math
import vtk

INPUT = "brain.vti"
OUTPUT = "brain_tensor_degeneracies.png"

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(INPUT)
reader.Update()
image = reader.GetOutput()
pd = image.GetPointData()
a, b, d = pd.GetArray("A"), pd.GetArray("B"), pd.GetArray("D")
dims, origin, spacing = image.GetDimensions(), image.GetOrigin(), image.GetSpacing()
nx, ny = dims[0], dims[1]

def tensor(i, j):
    k = j * nx + i
    return a.GetTuple1(k), b.GetTuple1(k), d.GetTuple1(k)

def field(i, j):
    aa, bb, dd = tensor(i, j)
    return aa - dd, 2.0 * bb

# Draw the major eigenvector as short, anisotropy-weighted line glyphs.
glyph_pts, glyph_lines, glyph_colors = vtk.vtkPoints(), vtk.vtkCellArray(), vtk.vtkUnsignedCharArray()
glyph_colors.SetNumberOfComponents(3)
for j in range(1, ny - 1, 2):
    for i in range(1, nx - 1, 2):
        aa, bb, dd = tensor(i, j)
        mag = math.hypot(aa - dd, 2 * bb)
        if mag < 1e-5:
            continue
        angle = 0.5 * math.atan2(2 * bb, aa - dd)
        length = min(0.85, 0.28 + 0.55 * math.sqrt(mag / 5e-4))
        x, y = origin[0] + i * spacing[0], origin[1] + j * spacing[1]
        p0 = glyph_pts.InsertNextPoint(x - length * math.cos(angle), y - length * math.sin(angle), 0)
        p1 = glyph_pts.InsertNextPoint(x + length * math.cos(angle), y + length * math.sin(angle), 0)
        glyph_lines.InsertNextCell(2); glyph_lines.InsertCellPoint(p0); glyph_lines.InsertCellPoint(p1)
        # Cool blue field, brighter where anisotropic.
        q = max(0, min(1, mag / 2.5))
        glyph_colors.InsertNextTuple3(int(34 + 38*q), int(88 + 95*q), int(126 + 115*q))
glyph = vtk.vtkPolyData(); glyph.SetPoints(glyph_pts); glyph.SetLines(glyph_lines); glyph.GetCellData().SetScalars(glyph_colors)

# Each grid quad is split along its lower-left to upper-right diagonal.  This
# exactly matches a piecewise-linear interpolation on those triangles.
wedge_pts, tri_pts = vtk.vtkPoints(), vtk.vtkPoints()
seen = set()
def consider_triangle(corners):
    # f and g are affine on this triangle: their common zero is solved in
    # barycentric coordinates relative to corner 0.
    (i0,j0),(i1,j1),(i2,j2) = corners
    f0,g0 = field(i0,j0); f1,g1 = field(i1,j1); f2,g2 = field(i2,j2)
    # Exclude zero-valued background vertices: those form non-isolated,
    # undefined regions rather than tensor-field singularities.
    if min(math.hypot(f0,g0), math.hypot(f1,g1), math.hypot(f2,g2)) < 1e-9:
        return
    u1,u2, v1,v2 = f1-f0, f2-f0, g1-g0, g2-g0
    det = u1*v2-u2*v1
    if abs(det) < 1e-12: return
    l1 = (-f0*v2 + u2*g0) / det
    l2 = (-u1*g0 + f0*v1) / det
    l0 = 1-l1-l2
    eps = 1e-7
    if min(l0,l1,l2) < -eps or max(l0,l1,l2) > 1+eps: return
    x = origin[0] + (l0*i0 + l1*i1 + l2*i2)*spacing[0]
    y = origin[1] + (l0*j0 + l1*j1 + l2*j2)*spacing[1]
    key = (round(x,5), round(y,5))
    if key in seen: return
    seen.add(key)
    # det(J)>0 is an index +1/2 tensor singularity (wedge); det(J)<0 is a trisector.
    (wedge_pts if det > 0 else tri_pts).InsertNextPoint(x,y,0)

for j in range(ny-1):
    for i in range(nx-1):
        consider_triangle(((i,j),(i+1,j),(i+1,j+1)))
        consider_triangle(((i,j),(i+1,j+1),(i,j+1)))

def spheres(points, rgb):
    poly = vtk.vtkPolyData(); poly.SetPoints(points)
    source = vtk.vtkSphereSource(); source.SetRadius(1.0); source.SetThetaResolution(20); source.SetPhiResolution(14)
    g = vtk.vtkGlyph3D(); g.SetInputData(poly); g.SetSourceConnection(source.GetOutputPort()); g.Update()
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(g.GetOutputPort())
    actor = vtk.vtkActor(); actor.SetMapper(mapper); actor.GetProperty().SetColor(*rgb); actor.GetProperty().SetSpecular(0.35); actor.GetProperty().SetSpecularPower(20)
    return actor

ren = vtk.vtkRenderer(); ren.SetBackground(0.025,0.035,0.055)
gm = vtk.vtkPolyDataMapper(); gm.SetInputData(glyph); gm.ScalarVisibilityOff()
ga = vtk.vtkActor(); ga.SetMapper(gm); ga.GetProperty().SetColor(0.18,0.58,0.92); ga.GetProperty().SetLineWidth(1.25); ga.GetProperty().SetOpacity(0.82)
ren.AddActor(ga)
ren.AddActor(spheres(wedge_pts, (1,1,1)))
ren.AddActor(spheres(tri_pts, (1.0,0.30,0.62)))

window = vtk.vtkRenderWindow(); window.SetOffScreenRendering(1); window.SetSize(1600, 980); window.AddRenderer(ren)
ren.ResetCamera(); cam = ren.GetActiveCamera(); cam.ParallelProjectionOn(); cam.SetParallelScale(ny * 0.57)
window.Render()
writer = vtk.vtkPNGWriter(); writer.SetFileName(OUTPUT); writer.SetInputConnection(window.GetRGBACharPixelData(0,0,1599,979,1).GetProducerPort()) if False else None
# vtkWindowToImageFilter reliably captures an off-screen ParaView render.
capture = vtk.vtkWindowToImageFilter(); capture.SetInput(window); capture.SetInputBufferTypeToRGBA(); capture.ReadFrontBufferOff(); capture.Update()
writer.SetInputConnection(capture.GetOutputPort()); writer.Write()
print(f"wedge points: {wedge_pts.GetNumberOfPoints()}")
print(f"trisector points: {tri_pts.GetNumberOfPoints()}")
print(OUTPUT)
