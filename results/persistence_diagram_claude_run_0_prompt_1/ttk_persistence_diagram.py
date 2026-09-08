from paraview.simple import *

# Load the TTK plugin
LoadDistributedPlugin('TopologyToolKit', ns=globals())

# --- Load the scalar field ---
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

# --- Compute the persistence diagram (TTK uses the sublevel set filtration
#     by default, i.e. ascending scalar order) ---
diagram = TTKPersistenceDiagram(Input=reader)
diagram.ScalarField = ['POINTS', 'Scalars_']
diagram.UpdatePipeline()

# --- Persistence simplification: keep only pairs with Persistence >= 0.04 ---
simplified = Threshold(Input=diagram)
simplified.Scalars = ['CELLS', 'Persistence']
simplified.LowerThreshold = 0.04
simplified.UpperThreshold = 1e9
simplified.ThresholdMethod = 'Between'
simplified.UpdatePipeline()

# --- Keep only order 0 (minimum-saddle) and order 2 (saddle-maximum) pairs ---
# PairType: 0 = minimum-saddle, 1 = saddle-saddle, 2 = saddle-maximum
order0 = Threshold(Input=simplified)
order0.Scalars = ['CELLS', 'PairType']
order0.LowerThreshold = 0.0
order0.UpperThreshold = 0.0
order0.ThresholdMethod = 'Between'
order0.UpdatePipeline()

order2 = Threshold(Input=simplified)
order2.Scalars = ['CELLS', 'PairType']
order2.LowerThreshold = 2.0
order2.UpperThreshold = 2.0
order2.ThresholdMethod = 'Between'
order2.UpdatePipeline()

diagram02 = AppendDatasets(Input=[order0, order2])
diagram02.UpdatePipeline()

# --- Display the piecewise-linear persistence diagram (birth/death plot) ---
renderView = CreateView('RenderView')
renderView.ViewSize = [900, 900]

disp = Show(diagram02, renderView)
disp.Representation = 'Surface'
ColorBy(disp, ('CELLS', 'PairType'))
disp.SetScalarBarVisibility(renderView, True)
disp.LineWidth = 2.0

renderView.OrientationAxesVisibility = 0
renderView.InteractionMode = '2D'
renderView.CameraParallelProjection = 1
renderView.ResetCamera()

# Look straight down the Z axis so birth (x) vs death (y) is displayed flat
bounds = diagram02.GetDataInformation().GetBounds()
cx = 0.5 * (bounds[0] + bounds[1])
cy = 0.5 * (bounds[2] + bounds[3])
cz = 0.5 * (bounds[4] + bounds[5])
renderView.CameraFocalPoint = [cx, cy, cz]
renderView.CameraPosition = [cx, cy, cz + 1.0]
renderView.CameraViewUp = [0.0, 1.0, 0.0]
renderView.ResetCamera()
Render(renderView)

SaveScreenshot('/workspace/persistence_diagram.png', renderView, ImageResolution=[900, 900])

print("Number of simplified pairs (order 0 + 2):", diagram02.GetDataInformation().GetNumberOfCells())
print("Saved persistence diagram screenshot to /workspace/persistence_diagram.png")
