from paraview.simple import *
from paraview import servermanager
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

input_file = '/workspace/QMCPACK.vti'
out_png = '/workspace/QMCPACK_persistence_diagram_H0_H2_simplified_0.04.png'
out_vtp = '/workspace/QMCPACK_persistence_diagram_H0_H2_simplified_0.04.vtp'

reader = XMLImageDataReader(FileName=[input_file])

# Persistence simplification at the requested (relative) threshold.
simplified = TTKTopologicalSimplificationByPersistence(Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 0
simplified.PairType = 'Extremum-Saddle'

diagram = TTKPersistenceDiagram(Input=simplified)
diagram.ScalarField = ['POINTS', 'Scalars_']
diagram.Dimensions = 'All Dimensions'
diagram.UseAllCores = 1

UpdatePipeline(proxy=diagram)
data = servermanager.Fetch(diagram)
print('diagram points', data.GetNumberOfPoints(), 'cells', data.GetNumberOfCells())
print('point arrays', [data.GetPointData().GetArrayName(i) for i in range(data.GetPointData().GetNumberOfArrays())])
print('cell arrays', [data.GetCellData().GetArrayName(i) for i in range(data.GetCellData().GetNumberOfArrays())])
for i in range(data.GetCellData().GetNumberOfArrays()):
 a=data.GetCellData().GetArray(i)
 print('CELL', a.GetName(), a.GetRange())

# TTK's diagram output is an unstructured grid; write its raw result explicitly.
writer = vtkXMLUnstructuredGridWriter()
writer.SetFileName(out_vtp)
writer.SetInputData(data)
writer.Write()

# Diagram cells encode pair type: 0=H0, 1=H1, 2=H2.  Render 0 and 2 only.
h0 = Threshold(Input=diagram)
h0.Scalars = ['CELLS', 'PairType']
h0.LowerThreshold = 0
h0.UpperThreshold = 0
h0.ThresholdMethod = 'Between'
h2 = Threshold(Input=diagram)
h2.Scalars = ['CELLS', 'PairType']
h2.LowerThreshold = 2
h2.UpperThreshold = 2
h2.ThresholdMethod = 'Between'

view = CreateView('RenderView')
view.ViewSize = [1500, 1100]
view.Background = [1, 1, 1]
view.BackgroundColorMode = 'Single Color'
view.OrientationAxesVisibility = 0

for source, color, label in [(h0, [0.12, 0.36, 0.78], 'H0'), (h2, [0.82, 0.18, 0.20], 'H2')]:
 display=Show(source, view)
 display.Representation = 'Points'
 display.PointSize = 9
 display.DiffuseColor = color
 display.AmbientColor = color
 display.Ambient = 1.0
 display.Diffuse = 0.0

view.AxesGrid.Visibility = 1
view.AxesGrid.XTitle = 'Birth'
view.AxesGrid.YTitle = 'Death'
view.AxesGrid.XTitleFontSize = 18
view.AxesGrid.YTitleFontSize = 18
view.AxesGrid.XLabelFontSize = 14
view.AxesGrid.YLabelFontSize = 14
view.AxesGrid.GridColor = [0.75, 0.75, 0.75]

diagonal = Line(Point1=[0, 0, 0], Point2=[0.8222, 0.8222, 0])
diagonal_display = Show(diagonal, view)
diagonal_display.DiffuseColor = [0.78, 0.78, 0.78]
diagonal_display.AmbientColor = [0.78, 0.78, 0.78]
diagonal_display.Ambient = 1.0
diagonal_display.Diffuse = 0.0
diagonal_display.LineWidth = 2.0

# The diagram is embedded in birth-death coordinates; use a top view.
view.CameraPosition = [0, 0, 1]
view.CameraFocalPoint = [0, 0, 0]
view.CameraViewUp = [0, 1, 0]
view.ResetCamera()

title = Text(registrationName='Title')
title.Text = 'Persistence diagram — piecewise-linear sub-level filtration'
td = Show(title, view)
td.WindowLocation = 'Upper Center'
td.FontSize = 22
td.Color = [0.05, 0.05, 0.05]

caption = Text(registrationName='Caption')
caption.Text = 'Persistence simplification = 0.04 (relative)     blue: H0     red: H2'
cd = Show(caption, view)
cd.WindowLocation = 'Lower Center'
cd.FontSize = 16
cd.Color = [0.05, 0.05, 0.05]

Render(view)
SaveScreenshot(out_png, view, ImageResolution=[1500, 1100])
print('wrote', out_png, out_vtp)
