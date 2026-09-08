from paraview.simple import *
import numpy as np
from paraview import servermanager as sm
from vtk.numpy_interface import dataset_adapter as dsa

paraview.simple._DisableFirstRenderCameraReset()

# --- Load data ---
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

di = reader.GetDataInformation()
arrays = [di.GetPointDataInformation().GetArrayInformation(i).GetName()
          for i in range(di.GetPointDataInformation().GetNumberOfArrays())]
print("Point data arrays:", arrays)
scalar_name = arrays[0]
print("Using scalar array:", scalar_name)

# --- Persistence diagram ---
pd = TTKPersistenceDiagram(Input=reader)
pd.ScalarField = ['POINTS', scalar_name]
pd.UpdatePipeline()

# --- Threshold the diagram on Persistence (keep pairs with persistence >= 0.04) ---
thresh = Threshold(Input=pd)
thresh.Scalars = ['CELLS', 'Persistence']
thresh.UpperThreshold = 0.04
thresh.ThresholdMethod = 'Above Upper Threshold'

thresh.UpdatePipeline()

# --- Topological simplification using the thresholded diagram ---
simp = TTKTopologicalSimplification(Domain=reader, Constraints=thresh)
simp.ScalarField = ['POINTS', scalar_name]
simp.UpdatePipeline()

# --- Critical points of the simplified field ---
cp = TTKScalarFieldCriticalPoints(Input=simp)
cp.ScalarField = ['POINTS', scalar_name]
cp.UpdatePipeline()

n_cp = cp.GetDataInformation().GetNumberOfPoints()
print("Number of critical points after simplification:", n_cp)

# --- Compute 4 isovalues splitting the (simplified) scalar field into 5 equal-count regions ---
simp_data = sm.Fetch(simp)
simp_np = dsa.WrapDataObject(simp_data)
vals = np.asarray(simp_np.PointData[scalar_name]).ravel()
qs = np.percentile(vals, [20, 40, 60, 80])
print("Isovalues (20/40/60/80 percentiles):", qs)

contour = Contour(Input=simp)
contour.ContourBy = ['POINTS', scalar_name]
contour.Isosurfaces = list(qs)
contour.UpdatePipeline()

# --- Render ---
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1600, 1200]
view.OrientationAxesVisibility = 0
view.Background = [0.15, 0.15, 0.15]

# Outline of dataset for context
outline = Outline(Input=reader)
outlineDisp = Show(outline, view)
outlineDisp.DiffuseColor = [1, 1, 1]

# Isocontours - translucent
contourDisp = Show(contour, view)
ColorBy(contourDisp, ('POINTS', scalar_name))
contourDisp.SetRepresentationType('Surface')
contourDisp.Opacity = 0.25
contour_LUT = GetColorTransferFunction(scalar_name)
contour_LUT.ApplyPreset('Cool to Warm', True)
HideScalarBarIfNotNeeded(contour_LUT, view)

# Critical points colored by CriticalType
cpDisp = Show(cp, view)
cpDisp.SetRepresentationType('Points')
cpDisp.PointSize = 14
cpDisp.RenderPointsAsSpheres = 1

ColorBy(cpDisp, ('POINTS', 'CriticalType'))
ctLUT = GetColorTransferFunction('CriticalType')
ctLUT.InterpretValuesAsCategories = 1
ctLUT.AnnotationsInitialized = 1

# 3D critical types: 0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum
ctLUT.Annotations = ['0', 'Minimum', '1', '1-Saddle', '2', '2-Saddle', '3', 'Maximum']
ctLUT.IndexedColors = [
    0.0, 0.4, 1.0,   # 0 minimum -> blue
    1.0, 1.0, 1.0,   # 1 1-saddle -> white
    1.0, 0.55, 0.0,  # 2 2-saddle -> orange
    1.0, 0.0, 0.0,   # 3 maximum -> red
]
cpDisp.SetScalarBarVisibility(view, True)

ResetCamera(view)
view.CameraPosition = view.CameraPosition
Render(view)

SaveScreenshot('/workspace/qmcpack_ttk_result.png', view, ImageResolution=[1600, 1200])
print("Saved screenshot to /workspace/qmcpack_ttk_result.png")
