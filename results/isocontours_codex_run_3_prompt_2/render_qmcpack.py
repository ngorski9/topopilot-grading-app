import vtk
from vtk.util.numpy_support import vtk_to_numpy
from topologytoolkit.ttkTopologicalSimplificationByPersistence import ttkTopologicalSimplificationByPersistence
from topologytoolkit.ttkScalarFieldCriticalPoints import ttkScalarFieldCriticalPoints

src = '/workspace/QMCPACK.vti'
out_vti = '/workspace/QMCPACK_simplified_0.04.vti'
out_vtp = '/workspace/QMCPACK_critical_points_0.04.vtp'
out_png = '/workspace/QMCPACK_simplified_0.04_visualization.png'

reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(src)

# Persistence-based simplification on the normalized scalar field.
simp = ttkTopologicalSimplificationByPersistence()
simp.SetInputConnection(reader.GetOutputPort())
simp.SetInputArrayToProcess(0, 0, 0, 0, 'Scalars_')
simp.SetPersistenceThreshold(0.04)
simp.SetThresholdIsAbsolute(True)
simp.Update()

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(out_vti)
writer.SetInputData(simp.GetOutput())
writer.Write()

# Extract all PL critical points from the simplified field.
critical = ttkScalarFieldCriticalPoints()
critical.SetInputConnection(simp.GetOutputPort())
critical.SetInputArrayToProcess(0, 0, 0, 0, 'Scalars_')
critical.Update()
cp = critical.GetOutput()
cp_writer = vtk.vtkXMLPolyDataWriter()
cp_writer.SetFileName(out_vtp)
cp_writer.SetInputData(cp)
cp_writer.Write()

# Four scalar quantiles create five equal-count scalar regions.
values = vtk_to_numpy(simp.GetOutput().GetPointData().GetArray('Scalars_'))
values.sort()
iso_values = [float(values[int((len(values)-1) * q / 5)]) for q in range(1, 5)]

contours = vtk.vtkContourFilter()
contours.SetInputData(simp.GetOutput())
contours.SetInputArrayToProcess(0, 0, 0, 0, 'Scalars_')
for i, value in enumerate(iso_values):
    contours.SetValue(i, value)
contours.Update()

contour_mapper = vtk.vtkPolyDataMapper()
contour_mapper.SetInputConnection(contours.GetOutputPort())
contour_mapper.SetScalarModeToUseCellData()
contour_mapper.ScalarVisibilityOff()
contour_actor = vtk.vtkActor()
contour_actor.SetMapper(contour_mapper)
contour_actor.GetProperty().SetColor(0.12, 0.68, 0.70)
contour_actor.GetProperty().SetOpacity(0.30)
contour_actor.GetProperty().SetInterpolationToPhong()

# Glyph the point set so every critical point remains visible in the 3-D view.
sphere = vtk.vtkSphereSource()
sphere.SetRadius(1.35)
sphere.SetThetaResolution(18)
sphere.SetPhiResolution(14)
glyph = vtk.vtkGlyph3D()
glyph.SetInputData(cp)
glyph.SetSourceConnection(sphere.GetOutputPort())
glyph.ScalingOff()
glyph.Update()

# TTK critical-type convention: 0=min, 1=1-saddle, 2=2-saddle, 3=max, 4=degenerate.
colors = {0:(0.08,0.25,1.0), 1:(1.0,1.0,1.0), 2:(1.0,0.45,0.02),
          3:(0.92,0.05,0.05), 4:(0.62,0.20,0.78)}
actors = []
for ctype, color in colors.items():
    threshold = vtk.vtkThresholdPoints()
    threshold.SetInputData(glyph.GetOutput())
    threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, 'CriticalType')
    threshold.ThresholdBetween(ctype, ctype)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(threshold.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetSpecular(0.55)
    actor.GetProperty().SetSpecularPower(35)
    actors.append(actor)

renderer = vtk.vtkRenderer()
renderer.SetBackground(0.035, 0.045, 0.075)
renderer.AddActor(contour_actor)
for actor in actors: renderer.AddActor(actor)

legend = vtk.vtkLegendBoxActor()
legend.SetNumberOfEntries(5)
for i, (label, color) in enumerate([('maximum',colors[3]), ('2-saddle',colors[2]), ('1-saddle',colors[1]), ('minimum',colors[0]), ('degenerate',colors[4])]):
    legend.SetEntryString(i, label)
    legend.SetEntryColor(i, color)
    legend.SetEntrySymbol(i, sphere.GetOutput())
legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
legend.GetPositionCoordinate().SetValue(0.015, 0.015)
legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
legend.GetPosition2Coordinate().SetValue(0.18, 0.20)
renderer.AddActor2D(legend)

text = vtk.vtkTextActor()
text.SetInput('Persistence simplification = 0.04 | 4 equal-count isocontours')
text.GetTextProperty().SetFontSize(22)
text.GetTextProperty().SetColor(0.92, 0.94, 1.0)
text.SetPosition(20, 1165)
renderer.AddActor2D(text)

window = vtk.vtkRenderWindow()
window.SetOffScreenRendering(1)
window.SetSize(1600, 1200)
window.AddRenderer(renderer)
renderer.ResetCamera()
cam = renderer.GetActiveCamera()
cam.Azimuth(32)
cam.Elevation(25)
cam.Zoom(1.18)
renderer.ResetCameraClippingRange()
window.Render()

png = vtk.vtkPNGWriter()
png.SetFileName(out_png)
grab = vtk.vtkWindowToImageFilter()
grab.SetInput(window)
grab.SetInputBufferTypeToRGBA()
grab.ReadFrontBufferOff()
grab.Update()
png.SetInputConnection(grab.GetOutputPort())
png.Write()

type_array = cp.GetPointData().GetArray('CriticalType')
counts = {i: 0 for i in range(5)}
for i in range(cp.GetNumberOfPoints()): counts[int(type_array.GetTuple1(i))] += 1
print('isovalues:', ', '.join(f'{x:.8g}' for x in iso_values))
print('critical point counts:', counts)
print(out_png)
