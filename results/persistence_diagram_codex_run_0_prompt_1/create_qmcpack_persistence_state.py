from paraview.simple import *

# Register TTK filters in ParaView's Python environment.
import topologytoolkit.ttkPersistenceDiagram
import topologytoolkit.ttkTopologicalSimplificationByPersistence

paraview.simple._DisableFirstRenderCameraReset()

reader = XMLImageDataReader(registrationName='QMCPACK scalar field',
                            FileName=['/workspace/QMCPACK.vti'])
reader.PointArrayStatus = ['Scalars_']

# Persistence-based scalar-field simplification (absolute value 0.04).
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.04)', Input=reader)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 1
simplified.PairType = 'Extremum-Saddle'
simplified.NumericalPerturbation = 0

# The TTK persistence diagram implements the sublevel-set filtration.
diagram = TTKPersistenceDiagram(
    registrationName='Sublevel persistence diagram', Input=simplified)
diagram.ScalarField = ['POINTS', 'Scalars_']
diagram.Dimensions = 'All Dimensions'
diagram.EmbedinDomain = 0
diagram.Backend = 'Discrete Morse Sandwich (IEEE TVCG 2023)'

# PairType is the homology order in a TTK persistence diagram.  Keep only
# type 0 (minimum--saddle) and type 2 (saddle--maximum) pairs.
order_0 = Threshold(registrationName='Persistence order 0', Input=diagram)
order_0.Scalars = ['CELLS', 'PairType']
order_0.ThresholdMethod = 'Between'
order_0.LowerThreshold = 0
order_0.UpperThreshold = 0
order_2 = Threshold(registrationName='Persistence order 2', Input=diagram)
order_2.Scalars = ['CELLS', 'PairType']
order_2.ThresholdMethod = 'Between'
order_2.LowerThreshold = 2
order_2.UpperThreshold = 2
orders_0_and_2 = AppendDatasets(registrationName='Orders 0 and 2',
                                Input=[order_0, order_2])

view = CreateView('RenderView')
view.ViewSize = [1100, 800]
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0

display = Show(orders_0_and_2, view, 'GeometryRepresentation')
display.Representation = 'Surface'
display.LineWidth = 2.5
display.DiffuseColor = [0.12, 0.32, 0.75]

view.ResetCamera()
view.CameraParallelProjection = 1
Render(view)

SaveScreenshot('/workspace/QMCPACK_persistence_diagram.png', view,
               ImageResolution=[1100, 800])
SaveState('/workspace/QMCPACK_persistence_diagram.pvsm')
