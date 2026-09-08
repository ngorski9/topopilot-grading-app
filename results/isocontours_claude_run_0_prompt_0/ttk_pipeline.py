from paraview.simple import *
import numpy as np
import paraview

paraview.simple._DisableFirstRenderCameraReset()
LoadDistributedPlugin('TopologyToolKit', ns=globals())

reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.PointArrayStatus = ['Scalars_']

# --- Persistence simplification (threshold 0.04, absolute) ---
simplify = TTKTopologicalSimplificationByPersistence(Input=reader)
simplify.InputArray = ['POINTS', 'Scalars_']
simplify.PersistenceThreshold = 0.04
simplify.ThresholdIsAbsolute = 1
simplify.UpdatePipeline()

# --- Critical points on simplified field ---
cp = TTKScalarFieldCriticalPoints(Input=simplify)
cp.ScalarField = ['POINTS', 'Scalars_']
cp.UpdatePipeline()
cpData = servermanager.Fetch(cp)

import vtk
def get_point_array_names(data):
    names = []
    if hasattr(data, 'GetPointData'):
        pdns = data.GetPointData()
        for i in range(pdns.GetNumberOfArrays()):
            names.append(pdns.GetArrayName(i))
    return names

if cpData.IsA('vtkMultiBlockDataSet'):
    it = cpData.NewIterator()
    it.InitTraversal()
    block = it.GetCurrentData()
    names = get_point_array_names(block)
else:
    names = get_point_array_names(cpData)
print("CriticalPoints point arrays:", names)

numPts = cpData.GetNumberOfPoints() if not cpData.IsA('vtkMultiBlockDataSet') else block.GetNumberOfPoints()
print("Number of critical points:", numPts)

# --- Isosurfaces: 4 isovalues dividing volume into 5 equal-sized regions ---
simpData = servermanager.Fetch(simplify)
if simpData.IsA('vtkMultiBlockDataSet'):
    it2 = simpData.NewIterator()
    it2.InitTraversal()
    simpBlock = it2.GetCurrentData()
else:
    simpBlock = simpData

from vtk.util.numpy_support import vtk_to_numpy
scalars = vtk_to_numpy(simpBlock.GetPointData().GetArray('Scalars_'))
scalars_sorted = np.sort(scalars)
quantile_fracs = [0.2, 0.4, 0.6, 0.8]
isovalues = [float(np.quantile(scalars_sorted, q)) for q in quantile_fracs]
print("Equal-volume isovalues:", isovalues)

contour = Contour(Input=simplify)
contour.ContourBy = ['POINTS', 'Scalars_']
contour.Isosurfaces = isovalues

# --- Rendering ---
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1400, 1000]
view.Background = [1, 1, 1]

contourDisplay = Show(contour, view)
contourDisplay.Representation = 'Surface'
contourDisplay.Opacity = 0.25
contourDisplay.DiffuseColor = [0.6, 0.6, 0.6]
ColorBy(contourDisplay, None)

cpDisplay = Show(cp, view)
cpDisplay.Representation = 'Points'
cpDisplay.PointSize = 12
cpDisplay.RenderPointsAsSpheres = 1

critArrayName = None
for cand in ['CriticalType', 'ttkVertexScalarField', 'VertexType']:
    if cand in names:
        critArrayName = cand
        break
if critArrayName is None:
    critArrayName = names[0]
print("Using critical type array:", critArrayName)

ColorBy(cpDisplay, ('POINTS', critArrayName))

lut = GetColorTransferFunction(critArrayName)
lut.InterpretValuesAsCategories = 1
lut.AnnotationsInitialized = 1
lut.Annotations = ['0', 'minima', '1', '1-saddle', '2', '2-saddle', '3', 'maxima']
lut.IndexedColors = [
    0.0, 0.0, 1.0,   # minima -> blue
    1.0, 1.0, 1.0,   # 1-saddle -> white
    1.0, 0.5, 0.0,   # 2-saddle -> orange
    1.0, 0.0, 0.0,   # maxima -> red
]

HideScalarBarIfNotNeeded(lut, view)

view.ResetCamera()
view.OrientationAxesVisibility = 0
Render(view)

SaveScreenshot('/workspace/ttk_result.png', view, ImageResolution=[1400, 1000])
print("Saved screenshot to /workspace/ttk_result.png")
