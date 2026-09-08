from paraview.simple import *
import numpy as np
from paraview import servermanager as sm

paraview.simple._DisableFirstRenderCameraReset()

ARRAY = "Scalars_"
PERSISTENCE_THRESHOLD = 0.04

reader = XMLImageDataReader(FileName=["/workspace/QMCPACK.vti"])
reader.PointArrayStatus = [ARRAY]

# --- Persistence Diagram ---
pd = TTKPersistenceDiagram(Input=reader)
pd.ScalarField = ARRAY
pd.InputOffsetField = ARRAY

# --- Threshold the diagram: keep pairs with Persistence >= threshold ---
pdThresh = Threshold(Input=pd)
pdThresh.Scalars = ['POINTS', 'Persistence']
pdThresh.LowerThreshold = PERSISTENCE_THRESHOLD
pdThresh.UpperThreshold = 1e12
pdThresh.ThresholdMethod = 'Above Upper Threshold' if False else 'Between'

# --- Topological Simplification using the persistence-thresholded diagram ---
simplify = TTKTopologicalSimplification(Domain=reader, Constraints=pdThresh)
simplify.ScalarField = ARRAY
simplify.InputOffsetField = ARRAY

# --- Recompute critical points on the simplified field ---
cp = TTKScalarFieldCriticalPoints(Input=simplify)
cp.ScalarField = ARRAY

# Fetch critical points data to build colored point clouds by CriticalType
cp.UpdatePipeline()
cpData = sm.Fetch(cp)

from vtk.util.numpy_support import vtk_to_numpy
import vtk

def merge_blocks(data):
    if data.IsA("vtkMultiBlockDataSet"):
        it = data.NewIterator()
        it.InitTraversal()
        blocks = []
        while not it.IsDoneWithTraversal():
            blk = it.GetCurrentDataObject()
            if blk is not None and blk.GetNumberOfPoints() > 0:
                blocks.append(blk)
            it.GoToNextItem()
        return blocks
    else:
        return [data]

blocks = merge_blocks(cpData)
points_list = []
types_list = []
for blk in blocks:
    pts = vtk_to_numpy(blk.GetPoints().GetData())
    ct = vtk_to_numpy(blk.GetPointData().GetArray("CriticalType"))
    points_list.append(pts)
    types_list.append(ct)
allPts = np.vstack(points_list)
allTypes = np.concatenate(types_list)

print("Critical point counts by type:", {int(t): int((allTypes==t).sum()) for t in np.unique(allTypes)})

# CriticalType convention (3D, TTK): 0=min,1=1-saddle,2=2-saddle,3=max
type_color = {0:(0,0,1), 1:(1,1,1), 2:(1,0.5,0), 3:(1,0,0)}
type_name = {0:"minima", 1:"1-saddles", 2:"2-saddles", 3:"maxima"}

# Save simplified critical points to disk grouped by type, reload as separate sources for coloring
import os
os.makedirs("/workspace/cp_split", exist_ok=True)

sources = {}
for t in sorted(np.unique(allTypes)):
    mask = allTypes == t
    sub_pts = allPts[mask]
    poly = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    for p in sub_pts:
        vpts.InsertNextPoint(p)
    poly.SetPoints(vpts)
    verts = vtk.vtkCellArray()
    for i in range(sub_pts.shape[0]):
        verts.InsertNextCell(1, [i])
    poly.SetVerts(verts)
    fname = f"/workspace/cp_split/type_{int(t)}.vtp"
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(fname)
    w.SetInputData(poly)
    w.Write()
    sources[int(t)] = fname

# --- Compute 4 isovalues splitting volume into 5 equal-sized regions (by voxel count) on simplified field ---
simplify.UpdatePipeline()
simplifiedData = sm.Fetch(simplify)
scal = vtk_to_numpy(simplifiedData.GetPointData().GetArray(ARRAY))
qs = np.percentile(scal, [20, 40, 60, 80])
isovalues = sorted(set(qs.tolist()))
print("Isovalues (20/40/60/80 percentile):", isovalues)

contour = Contour(Input=simplify)
contour.ContourBy = ['POINTS', ARRAY]
contour.Isosurfaces = isovalues
contour.PointMergeMethod = 'Uniform Binning'

# ============ Rendering ============
renderView1 = GetActiveViewOrCreate('RenderView')

contourDisplay = Show(contour, renderView1)
contourDisplay.Representation = 'Surface'
contourDisplay.ColorArrayName = ['POINTS', ARRAY]
contourDisplay.Opacity = 0.12
ctf = GetColorTransferFunction(ARRAY)
ctf.ApplyPreset('Cool to Warm', True)

readerDisplay = Show(reader, renderView1)
readerDisplay.Representation = 'Outline'
readerDisplay.AmbientColor = [1,1,1]
readerDisplay.DiffuseColor = [1,1,1]

for t, fname in sources.items():
    if t not in type_color:
        print(f"Skipping critical type {t} (not one of min/1-saddle/2-saddle/max) - {(allTypes==t).sum()} points")
        continue
    src = XMLPolyDataReader(FileName=[fname])
    disp = Show(src, renderView1)
    disp.Representation = 'Points'
    disp.PointSize = 14
    disp.RenderPointsAsSpheres = True
    disp.AmbientColor = list(type_color[t])
    disp.DiffuseColor = list(type_color[t])
    disp.ColorArrayName = [None, '']

renderView1.ResetCamera()
renderView1.Background = [0.05, 0.05, 0.07]
renderView1.OrientationAxesVisibility = 0
renderView1.CameraPosition = [-350, -350, 250]
renderView1.CameraFocalPoint = [34, 34, 57]
renderView1.CameraViewUp = [0, 0, 1]
renderView1.ResetCamera()
renderView1.CameraViewAngle = 30

SaveScreenshot("/workspace/qmcpack_persistence_simplified.png", renderView1, ImageResolution=[1600, 1200])
print("Saved screenshot to /workspace/qmcpack_persistence_simplified.png")
