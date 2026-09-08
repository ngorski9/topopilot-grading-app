from paraview.simple import *

paraview.simple._DisableFirstRenderCameraReset()

# Scalar field and persistence-based simplification (absolute value 0.04).
field = XMLImageDataReader(registrationName='QMCPACK scalar field', FileName=['/workspace/QMCPACK.vti'])
field.PointArrayStatus = ['Scalars_']
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.04)', Input=field)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1

# PL persistence diagram under the sublevel-set convention.
diagram = TTKPersistenceDiagram(registrationName='PL persistence diagram (sublevel)', Input=simplified)
diagram.ScalarField = ['POINTS', 'Scalars_']
diagram.Dimensions = 'All Dimensions'
diagram.EmbedinDomain = 0

# Keep the two requested homology orders from the PairType cell-data array.
order0 = Threshold(registrationName='Order 0', Input=diagram)
order0.Scalars = ['CELLS', 'PairType']
order0.ThresholdMethod = 'Between'
order0.LowerThreshold = -0.1
order0.UpperThreshold = 0.1
order2 = Threshold(registrationName='Order 2', Input=diagram)
order2.Scalars = ['CELLS', 'PairType']
order2.ThresholdMethod = 'Between'
order2.LowerThreshold = 1.9
order2.UpperThreshold = 2.1

view = CreateView('RenderView')
view.ViewSize = [1400, 1000]
view.Background = [1.0, 1.0, 1.0]
view.Background2 = [1.0, 1.0, 1.0]
view.UseGradientBackground = 0
view.OrientationAxesVisibility = 0
view.InteractionMode = '2D'
view.CameraParallelProjection = 1

o0display = Show(order0, view)
o0display.Representation = 'Surface'
o0display.DiffuseColor = [0.12, 0.36, 0.78]
o0display.LineWidth = 3.5
o2display = Show(order2, view)
o2display.Representation = 'Surface'
o2display.DiffuseColor = [0.85, 0.16, 0.18]
o2display.LineWidth = 3.5

# The diagonal is included as the conventional zero-persistence reference.
diagonal = Line(registrationName='Zero-persistence diagonal')
diagonal.Point1 = [0.0, 0.0, 0.0]
diagonal.Point2 = [2.05, 2.05, 0.0]
diagonal.Resolution = 1
diagdisplay = Show(diagonal, view)
diagdisplay.DiffuseColor = [0.35, 0.35, 0.35]
diagdisplay.LineWidth = 2.0

title = Text(registrationName='Diagram title')
title.Text = 'Piecewise-linear persistence diagram  |  Sublevel-set filtration  |  Simplification = 0.04'
titledisplay = Show(title, view)
titledisplay.WindowLocation = 'Upper Center'
titledisplay.FontSize = 20
titledisplay.Color = [0.05, 0.05, 0.05]

legend = Text(registrationName='Order legend')
legend.Text = 'Blue: order 0     Red: order 2     Gray: zero-persistence diagonal'
legenddisplay = Show(legend, view)
legenddisplay.WindowLocation = 'Lower Center'
legenddisplay.FontSize = 16
legenddisplay.Color = [0.08, 0.08, 0.08]

view.CameraPosition = [1.0, 1.0, 5.0]
view.CameraFocalPoint = [1.0, 1.0, 0.0]
view.CameraViewUp = [0.0, 1.0, 0.0]
view.CameraParallelScale = 1.15

Render(view)
SaveScreenshot('/workspace/QMCPACK_persistence_diagram_orders_0_2.png', view, ImageResolution=[1400, 1000])
SaveState('/workspace/QMCPACK_persistence_diagram.pvsm')
