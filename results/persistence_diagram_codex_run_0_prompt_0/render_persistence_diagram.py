from paraview.simple import *

# Input scalar field and persistence-based topological simplification.
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.PointArrayStatus = ['Scalars_']
simplified = TTKTopologicalSimplificationByPersistence(Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
# Interpret the requested threshold as 4% of the scalar range, TTK's standard
# persistence-simplification convention.
simplified.ThresholdIsAbsolute = 0

# Compute the sublevel-set persistence diagram of the simplified field.
diagram = TTKPersistenceDiagram(Input=simplified)
diagram.ScalarField = ['POINTS', 'Scalars_']

# TTK PairType 0 = minimum-saddle (order 0), 2 = saddle-maximum (order 2).
order0 = Threshold(Input=diagram)
order0.Scalars = ['CELLS', 'PairType']
order0.ThresholdMethod = 'Between'
order0.LowerThreshold = 0
order0.UpperThreshold = 0

order2 = Threshold(Input=diagram)
order2.Scalars = ['CELLS', 'PairType']
order2.ThresholdMethod = 'Between'
order2.LowerThreshold = 2
order2.UpperThreshold = 2

# Render the diagram directly in birth/death coordinates.
view = CreateView('RenderView')
view.ViewSize = [1500, 1050]
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.CameraParallelProjection = 1

for source, color in ((order0, [0.12, 0.40, 0.80]), (order2, [0.88, 0.22, 0.18])):
    display = Show(source, view)
    display.Representation = 'Surface'
    display.AmbientColor = color
    display.DiffuseColor = color
    display.LineWidth = 3.5

diagonal = Line(Point1=[-0.05, -0.05, 0.0], Point2=[1.05, 1.05, 0.0])
diagonal_display = Show(diagonal, view)
diagonal_display.AmbientColor = [0.55, 0.55, 0.55]
diagonal_display.DiffuseColor = [0.55, 0.55, 0.55]
diagonal_display.LineWidth = 1.5

# Birth and death coordinate axes.
view.CameraPosition = [0.5, 0.5, 2.0]
view.CameraFocalPoint = [0.5, 0.5, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelScale = 0.60

title = Text(registrationName='Title')
title.Text = 'Sublevel-set Persistence Diagram'
title_display = Show(title, view)
title_display.WindowLocation = 'Upper Center'
title_display.Position = [0.35, 0.93]
title_display.FontSize = 24
title_display.Color = [0.08, 0.08, 0.08]

legend = Text(registrationName='Legend')
legend.Text = 'Simplification threshold: 0.04 (relative scalar range)\nBlue: order 0 (minimum–saddle)     Red: order 2 (saddle–maximum)\nDiagonal: zero persistence.'
legend_display = Show(legend, view)
legend_display.WindowLocation = 'Lower Left Corner'
legend_display.Position = [0.03, 0.035]
legend_display.FontSize = 15
legend_display.Color = [0.1, 0.1, 0.1]

# Axis labels are rendered as annotations to make the exported image portable.
xlabel = Text(registrationName='XLabel')
xlabel.Text = 'Birth'
xd = Show(xlabel, view); xd.WindowLocation = 'Lower Center'; xd.Position = [0.48, 0.01]; xd.FontSize = 20; xd.Color = [0.05, 0.05, 0.05]
ylabel = Text(registrationName='YLabel')
ylabel.Text = 'Death'
yd = Show(ylabel, view); yd.WindowLocation = 'Upper Left Corner'; yd.Position = [0.02, 0.52]; yd.FontSize = 20; yd.Color = [0.05, 0.05, 0.05]

Render(view)
SaveScreenshot('/workspace/QMCPACK_persistence_diagram_orders_0_2.png', view, ImageResolution=[1500, 1050])
SaveState('/workspace/QMCPACK_persistence_diagram_orders_0_2.pvsm')
print('order-0 cells:', order0.GetDataInformation().GetNumberOfCells())
print('order-2 cells:', order2.GetDataInformation().GetNumberOfCells())
