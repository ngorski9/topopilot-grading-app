from paraview.simple import *

LoadDistributedPlugin('TopologyToolKit', ns=globals())

SIMPLIFICATION_THRESHOLD = 0.04

# 1. Load the scalar field
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.PointArrayStatus = ['Scalars_']

# 2. Compute the initial persistence diagram (sublevel set filtration)
pd0 = TTKPersistenceDiagram(Input=reader)
pd0.ScalarField = ['POINTS', 'Scalars_']

# 3. Remove the global min-max pair (infinite persistence) before thresholding
criticalPairs = Threshold(Input=pd0)
criticalPairs.Scalars = ['CELLS', 'PairIdentifier']
criticalPairs.LowerThreshold = 0
criticalPairs.UpperThreshold = 10**9
criticalPairs.ThresholdMethod = 'Between'

# 4. Keep only pairs with persistence ABOVE the simplification threshold
#    -> these are the significant features to PRESERVE; they become the
#       "Constraints" fed to TopologicalSimplification, which then removes
#       (flattens out) every pair below the threshold.
persistentPairs = Threshold(Input=criticalPairs)
persistentPairs.Scalars = ['CELLS', 'Persistence']
persistentPairs.LowerThreshold = SIMPLIFICATION_THRESHOLD
persistentPairs.UpperThreshold = 10**9
persistentPairs.ThresholdMethod = 'Between'

# 5. Simplify the scalar field on the original domain, keeping only the
#    significant (persistence > threshold) critical point pairs
simplification = TTKTopologicalSimplification(Domain=reader, Constraints=persistentPairs)
simplification.ScalarField = ['POINTS', 'Scalars_']

# 6. Recompute the (piecewise-linear) persistence diagram on the simplified field
pd1 = TTKPersistenceDiagram(Input=simplification)
pd1.ScalarField = ['POINTS', 'Scalars_']
pd1.UpdatePipeline()

from paraview import servermanager as sm
data = sm.Fetch(pd1)
cd = data.GetCellData()
print("Simplified persistence diagram cell arrays:", [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())])
print("Number of persistence pairs after simplification:", data.GetNumberOfCells())

pairType = cd.GetArray('PairType')
if pairType is not None:
    from collections import Counter
    counts = Counter(int(pairType.GetValue(i)) for i in range(pairType.GetNumberOfTuples()))
    print("Pair counts by dimension/order:", dict(counts))

# 7. Isolate the order-0 (min-saddle) and order-2 (saddle-max) pairs for display
diagramOrders02 = Threshold(Input=pd1)
diagramOrders02.Scalars = ['CELLS', 'PairType']
diagramOrders02.ThresholdMethod = 'Between'
diagramOrders02.LowerThreshold = 0
diagramOrders02.UpperThreshold = 2
# Note: PairType==1 (saddle-saddle) will also pass 'Between 0 and 2'; select
# 0 and 2 explicitly via two separate thresholds combined, done below instead.

renderView = CreateView('RenderView')

# Order 0 pairs (minimum-saddle)
order0 = Threshold(Input=pd1)
order0.Scalars = ['CELLS', 'PairType']
order0.ThresholdMethod = 'Between'
order0.LowerThreshold = 0
order0.UpperThreshold = 0
disp0 = Show(order0, renderView)
ColorBy(disp0, ('CELLS', 'Persistence'))
disp0.SetRepresentationType('Wireframe')

# Order 2 pairs (saddle-maximum)
order2 = Threshold(Input=pd1)
order2.Scalars = ['CELLS', 'PairType']
order2.ThresholdMethod = 'Between'
order2.LowerThreshold = 2
order2.UpperThreshold = 2
disp2 = Show(order2, renderView)
ColorBy(disp2, ('CELLS', 'Persistence'))
disp2.SetRepresentationType('Wireframe')

renderView.ResetCamera()
renderView.InteractionMode = '2D'
renderView.CameraParallelProjection = 1
renderView.OrientationAxesVisibility = 0
ResetCamera(renderView)
renderView.ResetCamera()
# Look straight down the Z axis so birth (X) vs death (Y) reads as a 2D diagram
bounds = renderView.ResetCamera()
camera = GetActiveCamera()
fp = camera.GetFocalPoint()
pos = camera.GetPosition()
dist = ((pos[0]-fp[0])**2 + (pos[1]-fp[1])**2 + (pos[2]-fp[2])**2) ** 0.5
camera.SetPosition(fp[0], fp[1], fp[2] + dist)
camera.SetViewUp(0, 1, 0)
renderView.ResetCamera()
Render(renderView)

SaveScreenshot('/workspace/persistence_diagram_orders_0_2.png', renderView, ImageResolution=[1200, 900])
print("Saved screenshot to /workspace/persistence_diagram_orders_0_2.png")

SaveData('/workspace/persistence_diagram_simplified.vtu', proxy=pd1)
print("Saved simplified persistence diagram to /workspace/persistence_diagram_simplified.vtu")
