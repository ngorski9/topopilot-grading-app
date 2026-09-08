import paraview.simple
from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# Scalar field and persistence simplification (0.04 relative to the 0--1 range).
field = XMLImageDataReader(registrationName='QMCPACK scalar field',
                           FileName=['/workspace/QMCPACK.vti'])
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.04)', Input=field)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 0

# Sublevel persistence diagram. Pair types 0 and 2 are the requested orders.
diagram = TTKPersistenceDiagram(registrationName='Sublevel persistence diagram',
                                Input=simplified)
diagram.ScalarField = ['POINTS', 'Scalars_']
diagram.Dimensions = 'All Dimensions'

order0 = Threshold(registrationName='Order 0', Input=diagram)
order0.Scalars = ['CELLS', 'PairType']
order0.ThresholdMethod = 'Between'
order0.LowerThreshold = 0
order0.UpperThreshold = 0

order2 = Threshold(registrationName='Order 2', Input=diagram)
order2.Scalars = ['CELLS', 'PairType']
order2.ThresholdMethod = 'Between'
order2.LowerThreshold = 2
order2.UpperThreshold = 2

orders_0_and_2 = AppendDatasets(registrationName='Orders 0 and 2',
                                Input=[order0, order2])

view = CreateView('RenderView')
view.ViewSize = [1200, 900]
view.Background = [1.0, 1.0, 1.0]
view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 0

display = Show(orders_0_and_2, view, 'GeometryRepresentation')
ColorBy(display, ('CELLS', 'PairType'))
display.SetRepresentationType('Surface With Edges')
display.LineWidth = 4.0
display.PointSize = 8.0
lut = GetColorTransferFunction('PairType')
lut.InterpretValuesAsCategories = 1
lut.Annotations = ['0', 'Order 0', '2', 'Order 2']
lut.IndexedColors = [0.1216, 0.4667, 0.7059, 0.8392, 0.1529, 0.1569]
lut.IndexedOpacities = [1.0, 1.0]
display.LookupTable = lut
display.SetScalarBarVisibility(view, True)

title = Text(registrationName='Diagram title')
title.Text = 'Piecewise-linear persistence diagram — sublevel set\nPersistence simplification: 0.04 | Orders: 0 and 2'
title_display = Show(title, view)
title_display.WindowLocation = 'Upper Center'
title_display.FontSize = 18
title_display.Color = [0.05, 0.05, 0.05]

view.CameraPosition = [0.5, 0.5, 3.0]
view.CameraFocalPoint = [0.5, 0.5, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelScale = 0.62
Render(view)
SaveScreenshot('/workspace/QMCPACK_persistence_diagram.png', view)
SaveState('/workspace/QMCPACK_persistence_diagram.pvsm')
print('Saved /workspace/QMCPACK_persistence_diagram.png')
print('Saved /workspace/QMCPACK_persistence_diagram.pvsm')
