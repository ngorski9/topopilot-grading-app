from paraview.simple import *
import os

paraview.simple._DisableFirstRenderCameraReset()

DATA_DIR = "./cloud"
N_TIMESTEPS = 3
PERSISTENCE_THRESHOLD = 0.5
SPHERE_RADIUS = 2.0
files = [os.path.join(DATA_DIR, f"cloud{i}.vti") for i in range(1, N_TIMESTEPS + 1)]

# ---------------------------------------------------------------------------
# 1. Load the time-dependent dataset (first 3 timesteps) and render the
#    original scalar field.
# ---------------------------------------------------------------------------
reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

renderView1 = GetActiveViewOrCreate('RenderView')
renderView1.ViewSize = [1000, 800]

origDisplay = Show(reader, renderView1)
ColorBy(origDisplay, ('POINTS', 'Scalars_'))
origDisplay.SetScalarBarVisibility(renderView1, True)
origLUT = GetColorTransferFunction('Scalars_')
origLUT.ApplyPreset('Cold and Hot', True)
renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/original_scalar_field.png', renderView1, ImageResolution=[1000, 800])

# ---------------------------------------------------------------------------
# 2. Persistence simplification (persistence threshold = 0.5), applied via
#    TTK's Persistence Diagram + Topological Simplification pair.
# ---------------------------------------------------------------------------
persistenceDiagram = TTKPersistenceDiagram(Input=reader)
persistenceDiagram.ScalarField = ['POINTS', 'Scalars_']
persistenceDiagram.UpdatePipeline()

constraintThreshold = Threshold(Input=persistenceDiagram)
constraintThreshold.Scalars = ['CELLS', 'Persistence']
constraintThreshold.LowerThreshold = PERSISTENCE_THRESHOLD
constraintThreshold.UpperThreshold = 1e9
constraintThreshold.ThresholdMethod = 'Between'
constraintThreshold.UpdatePipeline()

simplification = TTKTopologicalSimplification(Domain=reader, Constraints=constraintThreshold)
simplification.ScalarField = ['POINTS', 'Scalars_']
simplification.VertexIdentifierField = ['POINTS', 'ttkVertexScalarField']
simplification.UpdatePipeline()

simplifiedDisplay = Show(simplification, renderView1)
ColorBy(simplifiedDisplay, ('POINTS', 'Scalars_'))
simplifiedDisplay.SetScalarBarVisibility(renderView1, True)
simplifiedLUT = GetColorTransferFunction('Scalars_')
simplifiedLUT.ApplyPreset('Cold and Hot', True)
Hide(reader, renderView1)
renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/simplified_scalar_field.png', renderView1, ImageResolution=[1000, 800])

# ---------------------------------------------------------------------------
# 3. Recompute persistence diagrams on the simplified field (one per
#    timestep), extract the piecewise-linear MAXIMA (saddle-max pairs with
#    persistence >= 0.5), and track them across the 3 timesteps by solving
#    an optimal-transport (earth mover's distance) assignment between the
#    maxima of consecutive timesteps with the Python Optimal Transport (POT)
#    library.
#
# NOTE: this environment's TTKTrackingFromPersistenceDiagrams filter
# segfaults inside libttkPersistenceDiagram.so's VTUToDiagram() when fed a
# multiblock of per-timestep diagrams (reproduced independently of how the
# multiblock is assembled - GroupTimeSteps, GroupDatasets, and a hand-built
# vtkMultiBlockDataSet all crash identically, while TTKBottleneckDistance
# consumes the very same diagrams without issue). We therefore compute the
# earth mover's distance matching ourselves with POT, using the exact same
# persistence diagrams TTK produces.
# ---------------------------------------------------------------------------
import vtk
import numpy as np
import ot
from paraview import servermanager as sm
from vtk.util import numpy_support as ns

persistenceDiagram2 = TTKPersistenceDiagram(Input=simplification)
persistenceDiagram2.ScalarField = ['POINTS', 'Scalars_']

maximaPerTimestep = []
for t in reader.TimestepValues:
    persistenceDiagram2.UpdatePipeline(t)
    diagram = sm.Fetch(persistenceDiagram2)

    coords = ns.vtk_to_numpy(diagram.GetPointData().GetArray('Coordinates'))
    criticalType = ns.vtk_to_numpy(diagram.GetPointData().GetArray('CriticalType'))
    pairType = ns.vtk_to_numpy(diagram.GetCellData().GetArray('PairType'))
    persistence = ns.vtk_to_numpy(diagram.GetCellData().GetArray('Persistence'))

    maxima = []
    for cellId in np.where(pairType == 1)[0]:
        if persistence[cellId] < PERSISTENCE_THRESHOLD:
            continue
        cell = diagram.GetCell(int(cellId))
        p0, p1 = cell.GetPointId(0), cell.GetPointId(1)
        maxPointId = p1 if criticalType[p1] == criticalType.max() else p0
        maxima.append((coords[maxPointId].copy(), float(persistence[cellId])))
    maximaPerTimestep.append(maxima)
    print(f"timestep t={t}: {len(maxima)} maxima with persistence >= {PERSISTENCE_THRESHOLD}")

# Match maxima between consecutive timesteps by solving an optimal
# transport problem (earth mover's distance) on their Euclidean positions.
tracks = []  # list of polylines, each a list of 3D points (one per timestep, possibly shorter)
activeTrack = {}  # index into maximaPerTimestep[t] -> track id
for t in range(len(maximaPerTimestep) - 1):
    curMax = maximaPerTimestep[t]
    nextMax = maximaPerTimestep[t + 1]
    if not curMax or not nextMax:
        continue

    curPos = np.array([m[0] for m in curMax])
    nextPos = np.array([m[0] for m in nextMax])
    curPersistence = [m[1] for m in curMax]
    nextPersistence = [m[1] for m in nextMax]

    # Uniform marginals; ground cost = squared Euclidean distance -> the
    # resulting optimal plan realizes the (squared) Wasserstein / earth
    # mover's distance between the two maxima point clouds.
    a = np.ones(len(curPos)) / len(curPos)
    b = np.ones(len(nextPos)) / len(nextPos)
    costMatrix = ot.dist(curPos, nextPos, metric='sqeuclidean')
    transportPlan = ot.emd(a, b, costMatrix)

    for i in range(len(curPos)):
        j = int(np.argmax(transportPlan[i]))
        if transportPlan[i, j] <= 0:
            continue
        trackId = activeTrack.get(i)
        if trackId is None:
            trackId = len(tracks)
            tracks.append([(curPos[i], curPersistence[i])])
        tracks[trackId].append((nextPos[j], nextPersistence[j]))
        activeTrack[j] = trackId
    activeTrack = {j: tid for j, tid in activeTrack.items() if j < len(nextPos)}

print(f"Tracked {len(tracks)} maxima trajectories across {N_TIMESTEPS} timesteps "
      f"(earth mover's distance matching).")

# ---------------------------------------------------------------------------
# 4. Build a VTK polydata (points + polylines) for the tracked maxima and
#    visualize them together with the original scalar field: spheres of
#    radius 2 colored with a warm-cold colormap.
# ---------------------------------------------------------------------------
points = vtk.vtkPoints()
lines = vtk.vtkCellArray()
values = vtk.vtkDoubleArray()
values.SetName('Scalars_')

for track in tracks:
    ids = []
    for p, persistenceValue in track:
        pid = points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
        values.InsertNextValue(persistenceValue)
        ids.append(pid)
    if len(ids) > 1:
        polyline = vtk.vtkPolyLine()
        polyline.GetPointIds().SetNumberOfIds(len(ids))
        for k, pid in enumerate(ids):
            polyline.GetPointIds().SetId(k, pid)
        lines.InsertNextCell(polyline)

trackPolyData = vtk.vtkPolyData()
trackPolyData.SetPoints(points)
trackPolyData.SetLines(lines)
trackPolyData.GetPointData().AddArray(values)
trackPolyData.GetPointData().SetActiveScalars('Scalars_')

trackProducer = TrivialProducer()
trackProducer.GetClientSideObject().SetOutput(trackPolyData)
trackProducer.UpdatePipeline()

Hide(simplification, renderView1)
Show(reader, renderView1)

trackDisplay = Show(trackProducer, renderView1)
trackDisplay.SetRepresentationType('Points')
trackDisplay.RenderPointsAsSpheres = True
trackDisplay.PointSize = SPHERE_RADIUS * 2
try:
    trackDisplay.SetRepresentationType('3D Glyphs')
    trackDisplay.GlyphType = 'Sphere'
    trackDisplay.GlyphType.Radius = SPHERE_RADIUS
except Exception:
    pass

ColorBy(trackDisplay, ('POINTS', 'Scalars_'))
trackLUT = GetColorTransferFunction('Scalars_')
trackLUT.ApplyPreset('Cold and Hot', True)
trackDisplay.SetScalarBarVisibility(renderView1, True)

renderView1.ResetCamera()
Render()
SaveScreenshot('/workspace/tracked_maxima.png', renderView1, ImageResolution=[1000, 800])

print("Pipeline complete. Screenshots written to /workspace/*.png")
