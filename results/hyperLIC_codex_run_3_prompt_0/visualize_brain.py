from paraview.simple import *
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataWriter
from vtkmodules.vtkCommonCore import vtkPoints, vtkIntArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray

INPUT = "/workspace/brain.vti"
WEDGE = "/workspace/brain_wedges.vtp"
TRISECTOR = "/workspace/brain_trisectors.vtp"


def make_points(filename, points):
    data = vtkPolyData()
    vtk_points = vtkPoints()
    cells = vtkCellArray()
    kinds = vtkIntArray()
    kinds.SetName("DegeneracyType")
    for x, y in points:
        point_id = vtk_points.InsertNextPoint(x, y, 0.0)
        cells.InsertNextCell(1)
        cells.InsertCellPoint(point_id)
        kinds.InsertNextValue(1)
    data.SetPoints(vtk_points)
    data.SetVerts(cells)
    data.GetPointData().AddArray(kinds)
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(data)
    writer.Write()


# A symmetric 2D tensor has a degeneracy where (A-D, 2B)=(0,0).  The input is
# piecewise linear, so solve this pair exactly on each triangle of every cell.
image_reader = vtkXMLImageDataReader()
image_reader.SetFileName(INPUT)
image_reader.Update()
image = image_reader.GetOutput()
extent = image.GetExtent()
origin = image.GetOrigin()
spacing = image.GetSpacing()
a = image.GetPointData().GetArray("A")
b = image.GetPointData().GetArray("B")
d = image.GetPointData().GetArray("D")
nx = extent[1] - extent[0] + 1
ny = extent[3] - extent[2] + 1


def val(array, i, j):
    return array.GetTuple1((j - extent[2]) * nx + (i - extent[0]))


wedges, trisectors = [], []
eps = 1.0e-10
for j in range(extent[2], extent[3]):
    for i in range(extent[0], extent[1]):
        # Consistent diagonal yields a simplicial (piecewise-linear) field.
        corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
        for tri in ((corners[0], corners[1], corners[2]),
                    (corners[0], corners[2], corners[3])):
            f = [val(a, *p) - val(d, *p) for p in tri]
            g = [2.0 * val(b, *p) for p in tri]
            # f = f0 + u(f1-f0) + v(f2-f0), likewise g.
            f1, f2 = f[1] - f[0], f[2] - f[0]
            g1, g2 = g[1] - g[0], g[2] - g[0]
            determinant = f1 * g2 - f2 * g1
            if abs(determinant) < eps:
                continue
            u = (-f[0] * g2 + f2 * g[0]) / determinant
            v = (-f1 * g[0] + f[0] * g1) / determinant
            w = 1.0 - u - v
            if u >= -eps and v >= -eps and w >= -eps:
                x = origin[0] + spacing[0] * (w * tri[0][0] + u * tri[1][0] + v * tri[2][0])
                y = origin[1] + spacing[1] * (w * tri[0][1] + u * tri[1][1] + v * tri[2][1])
                # Positive line-field index is a wedge; negative is a trisector.
                target = wedges if determinant > 0.0 else trisectors
                if not any((x-qx)**2 + (y-qy)**2 < 1e-12 for qx, qy in target):
                    target.append((x, y))

make_points(WEDGE, wedges)
make_points(TRISECTOR, trisectors)
print("Wedges:", len(wedges), "Trisectors:", len(trisectors))

field = XMLImageDataReader(registrationName="brain tensor field", FileName=[INPUT])
wedges_source = XMLPolyDataReader(registrationName="wedges", FileName=[WEDGE])
trisectors_source = XMLPolyDataReader(registrationName="trisectors", FileName=[TRISECTOR])

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1200, 750]
view.Background = [0.08, 0.08, 0.10]
field_display = Show(field, view)
field_display.DiffuseColor = [0.42, 0.48, 0.56]
field_display.Opacity = 0.72
field_display.Representation = "Surface"

for source, color, name in ((wedges_source, [1.0, 1.0, 1.0], "Wedges (radius 1)"),
                            (trisectors_source, [1.0, 0.41, 0.71], "Trisectors (radius 1)")):
    glyph = Glyph(registrationName=name, Input=source, GlyphType="Sphere")
    glyph.GlyphType.Radius = 1.0
    glyph.GlyphType.ThetaResolution = 16
    glyph.GlyphType.PhiResolution = 12
    glyph.ScaleArray = ["POINTS", "No scale array"]
    glyph.ScaleFactor = 1.0
    glyph.GlyphMode = "All Points"
    display = Show(glyph, view)
    display.DiffuseColor = color
    display.Specular = 0.35
    display.SpecularPower = 24.0

field_display.SetScalarBarVisibility(view, False)
view.CameraParallelProjection = 1
view.CameraPosition = [53.5, 32.5, 200.0]
view.CameraFocalPoint = [53.5, 32.5, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelScale = 72.0
Render()
SaveScreenshot("/workspace/brain_degeneracies.png", view, ImageResolution=[1200, 750])
SaveState("/workspace/brain_degeneracies.pvsm")
