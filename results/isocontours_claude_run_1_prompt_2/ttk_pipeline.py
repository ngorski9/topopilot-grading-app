from paraview.simple import *
import numpy as np
from paraview import servermanager as sm
from vtk.util.numpy_support import vtk_to_numpy

paraview.simple._DisableFirstRenderCameraReset()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
reader = XMLImageDataReader(FileName=['/workspace/QMCPACK.vti'])
reader.UpdatePipeline()

data0 = sm.Fetch(reader)
pd = data0.GetPointData()
array_name = pd.GetArrayName(0)
print("Using scalar array:", array_name)

# ---------------------------------------------------------------------------
# 2. Persistence diagram -> threshold at 0.04 -> topological simplification
# ---------------------------------------------------------------------------
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', array_name]
persistenceDiagram.UpdatePipeline()

# Keep only pairs with persistence >= 0.04 (threshold on the diagram output)
threshold = Threshold(Input=persistenceDiagram)
threshold.Scalars = ['CELLS', 'Persistence']
threshold.UpperThreshold = 0.04
threshold.ThresholdMethod = 'Above Upper Threshold'
threshold.UpdatePipeline()

simplification = TTKTopologicalSimplification(Domain=reader, Constraints=threshold)
simplification.ScalarField = ['POINTS', array_name]
simplification.UpdatePipeline()

simplified_array_name = array_name

# ---------------------------------------------------------------------------
# 3. Critical points of the simplified (persistence-simplified) field
# ---------------------------------------------------------------------------
criticalPoints = TTKScalarFieldCriticalPoints(Input=simplification)
criticalPoints.ScalarField = ['POINTS', simplified_array_name]
criticalPoints.UpdatePipeline()

cp_data = sm.Fetch(criticalPoints)
cp_pd = cp_data.GetPointData()
ctype = vtk_to_numpy(cp_pd.GetArray('CriticalType'))
print("Critical point counts by type:", {int(t): int((ctype == t).sum()) for t in np.unique(ctype)})

# ---------------------------------------------------------------------------
# 4. Compute 4 isovalues that divide the volume into 5 equal-volume regions
#    (equal voxel-count bins of the simplified scalar field)
# ---------------------------------------------------------------------------
simp_data = sm.Fetch(simplification)
simp_arr = vtk_to_numpy(simp_data.GetPointData().GetArray(simplified_array_name))
percentiles = [20, 40, 60, 80]
isovalues = list(np.percentile(simp_arr, percentiles))
print("Isovalues for 5 equal-volume regions:", isovalues)

# ---------------------------------------------------------------------------
# 5. Render setup
# ---------------------------------------------------------------------------
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1400, 1000]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.Background2 = [1, 1, 1]

# Isocontours dividing volume into 5 equal regions
contour = Contour(Input=simplification)
contour.ContourBy = ['POINTS', simplified_array_name]
contour.Isosurfaces = isovalues
contour.PointMergeMethod = 'Uniform Binning'
contourDisplay = Show(contour, view)
contourDisplay.Representation = 'Surface'
ColorBy(contourDisplay, ('POINTS', simplified_array_name))
contourLUT = GetColorTransferFunction(simplified_array_name)
contourLUT.ApplyPreset('Cool to Warm', True)
contourDisplay.Opacity = 0.35
contourDisplay.SetScalarBarVisibility(view, True)

outline = Outline(Input=reader)
outlineDisplay = Show(outline, view)
outlineDisplay.AmbientColor = [0.0, 0.0, 0.0]
outlineDisplay.DiffuseColor = [0.0, 0.0, 0.0]
outlineDisplay.LineWidth = 2.0

# Critical points: split by type and color explicitly
# Critical type convention (3D scalar field): 0=min, 1=1-saddle, 2=2-saddle, 3=max
type_colors = {
    0: [0.0, 0.0, 1.0],   # minima -> blue
    1: [1.0, 1.0, 1.0],   # 1-saddles -> white
    2: [1.0, 0.5, 0.0],   # 2-saddles -> orange
    3: [1.0, 0.0, 0.0],   # maxima -> red
}
type_names = {0: 'Minima', 1: '1-Saddles', 2: '2-Saddles', 3: 'Maxima'}

for t, color in type_colors.items():
    thr = Threshold(Input=criticalPoints)
    thr.Scalars = ['POINTS', 'CriticalType']
    thr.LowerThreshold = t
    thr.UpperThreshold = t
    thr.ThresholdMethod = 'Between'
    thr.UpdatePipeline()

    n_pts = thr.GetDataInformation().GetNumberOfPoints()
    if n_pts == 0:
        continue

    glyph = Glyph(Input=thr, GlyphType='Sphere')
    glyph.ScaleArray = ['POINTS', 'No scale array']
    glyph.ScaleFactor = 3.0
    glyph.GlyphMode = 'All Points'
    glyph.GlyphType.Radius = 1.0

    disp = Show(glyph, view)
    disp.Representation = 'Surface'
    disp.AmbientColor = color
    disp.DiffuseColor = color
    disp.ColorArrayName = [None, '']
    print(f"{type_names[t]}: {n_pts} points, color {color}")

view.ResetCamera()
view.CameraViewAngle = 30
Render(view)
SaveScreenshot('/workspace/qmcpack_ttk_critical_points.png', view, ImageResolution=[1400, 1000])
print("Saved screenshot to /workspace/qmcpack_ttk_critical_points.png")
