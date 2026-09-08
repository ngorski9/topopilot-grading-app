"""
TTK time-varying pipeline for the ./cloud dataset:
1. Load the .vti time series.
2. Show the original scalar field.
3. Persistence simplification (threshold 0.5) of the scalar field.
4. Compute persistence diagrams for 3 time steps, keep maxima (saddle-max pairs),
   and track them across time with TTKTrackingFromPersistenceDiagrams using the
   Wasserstein-1 (earth mover's distance) metric.
5. Visualize the scalar field + tracked maxima as spheres (radius 2), warm-cold colormap.
"""
import glob
import os
import re

import numpy as np
import ot
import vtkmodules.vtkCommonCore as vtkCore
import vtkmodules.vtkCommonDataModel as vtkDM
from paraview.simple import *
from paraview import servermanager as sm

DATA_DIR = "/workspace/cloud"
OUT_DIR = "/workspace/cloud_out"
os.makedirs(OUT_DIR, exist_ok=True)

SCALAR_ARRAY = "Scalars_"
PERSISTENCE_THRESHOLD = 0.5
N_TRACK_STEPS = 3
SPHERE_RADIUS = 2.0

# ---------------------------------------------------------------------------
# 1. Load the time-varying series (sorted numerically: cloud1, cloud2, ...)
# ---------------------------------------------------------------------------
files = glob.glob(os.path.join(DATA_DIR, "cloud*.vti"))
files.sort(key=lambda f: int(re.search(r"(\d+)", os.path.basename(f)).group(1)))
print("Loaded %d files, first 5: %s" % (len(files), files[:5]))

reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = [SCALAR_ARRAY]
reader.UpdatePipeline()

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1000, 800]

# ---------------------------------------------------------------------------
# 2. Show the original scalar field (first time step)
# ---------------------------------------------------------------------------
readerDisp = Show(reader, view)
ColorBy(readerDisp, ("POINTS", SCALAR_ARRAY))
readerDisp.SetScalarBarVisibility(view, True)
readerDisp.LookupTable = GetColorTransferFunction(SCALAR_ARRAY)
readerDisp.LookupTable.ApplyPreset("Cool to Warm", True)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "1_original_scalar_field.png"), view)
Hide(reader, view)

# ---------------------------------------------------------------------------
# 3. Persistence simplification (threshold 0.5) of the scalar field
# ---------------------------------------------------------------------------
simplified = TTKTopologicalSimplificationByPersistence(Input=reader)
simplified.InputArray = ["POINTS", SCALAR_ARRAY]
simplified.PersistenceThreshold = PERSISTENCE_THRESHOLD
simplified.ThresholdIsAbsolute = 1
simplified.PairType = "Extremum-Saddle"
simplified.UpdatePipeline()

simplifiedDisp = Show(simplified, view)
ColorBy(simplifiedDisp, ("POINTS", SCALAR_ARRAY))
simplifiedDisp.SetScalarBarVisibility(view, True)
simplifiedDisp.LookupTable = GetColorTransferFunction(SCALAR_ARRAY)
simplifiedDisp.LookupTable.ApplyPreset("Cool to Warm", True)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "2_simplified_scalar_field.png"), view)
Hide(simplified, view)

# ---------------------------------------------------------------------------
# 4. Track piecewise-linear maxima across N_TRACK_STEPS time steps
#    using earth mover's distance (Wasserstein-1)
# ---------------------------------------------------------------------------
timesteps = list(reader.TimestepValues)
if timesteps and isinstance(timesteps[0], list):
    timesteps = timesteps[0]
trackTimes = timesteps[:N_TRACK_STEPS]
print("Tracking time steps:", trackTimes)

# For each tracked time step, compute the persistence diagram of the
# simplified field and extract the maxima (saddle-max pairs, PairType == 1):
# their real spatial location ("Coordinates" point array) and persistence.
maximaPerFrame = []  # list of (Nx3 coords array, N persistence array)
for t in trackTimes:
    readerAtT = XMLImageDataReader(FileName=files)
    readerAtT.PointArrayStatus = [SCALAR_ARRAY]

    simplifiedAtT = TTKTopologicalSimplificationByPersistence(Input=readerAtT)
    simplifiedAtT.InputArray = ["POINTS", SCALAR_ARRAY]
    simplifiedAtT.PersistenceThreshold = PERSISTENCE_THRESHOLD
    simplifiedAtT.ThresholdIsAbsolute = 1
    simplifiedAtT.PairType = "Extremum-Saddle"

    diagramAtT = TTKPersistenceDiagram(Input=simplifiedAtT)
    diagramAtT.ScalarField = ["POINTS", SCALAR_ARRAY]
    diagramAtT.UpdatePipeline(t)

    data = sm.Fetch(diagramAtT)
    critType = data.GetPointData().GetArray("CriticalType")
    coordsArr = data.GetPointData().GetArray("Coordinates")
    pairType = data.GetCellData().GetArray("PairType")
    persistenceArr = data.GetCellData().GetArray("Persistence")

    coords, persistences = [], []
    for c in range(data.GetNumberOfCells()):
        if int(pairType.GetValue(c)) != 1:
            continue  # keep only saddle-maximum pairs
        cell = data.GetCell(c)
        p0, p1 = cell.GetPointId(0), cell.GetPointId(1)
        # the maximum endpoint has the larger CriticalType value
        maxPointId = p0 if critType.GetValue(p0) > critType.GetValue(p1) else p1
        coords.append(coordsArr.GetTuple3(maxPointId))
        persistences.append(persistenceArr.GetValue(c))

    coords = np.array(coords, dtype=float)
    persistences = np.array(persistences, dtype=float)

    # Keep only the most prominent maxima so the earth-mover's-distance
    # assignment problem (and the resulting visualization) stays tractable.
    TOP_K = 200
    if len(persistences) > TOP_K:
        keep = np.argsort(persistences)[::-1][:TOP_K]
        coords, persistences = coords[keep], persistences[keep]

    print("t=%s: tracking %d maxima" % (t, len(persistences)))
    maximaPerFrame.append((coords, persistences))

# Track the maxima across consecutive time steps using earth mover's distance
# (optimal transport, via the POT library) with squared Euclidean ground cost.
trackEdges = []  # list of (frameIndex, pointIndexInFrame, pointIndexInNextFrame)
for i in range(len(maximaPerFrame) - 1):
    coordsA, persA = maximaPerFrame[i]
    coordsB, persB = maximaPerFrame[i + 1]
    a = persA / persA.sum()
    b = persB / persB.sum()
    M = ot.dist(coordsA, coordsB, metric="sqeuclidean")
    G = ot.emd(a, b, M)  # earth mover's distance transport plan
    for row in range(G.shape[0]):
        col = int(np.argmax(G[row]))
        if G[row, col] > 0:
            trackEdges.append((i, row, col))

# ---------------------------------------------------------------------------
# Build a vtkPolyData with all tracked maxima (as points, colored by
# persistence) and polylines connecting matched maxima across time steps.
# ---------------------------------------------------------------------------
points = vtkCore.vtkPoints()
persistenceArrayOut = vtkCore.vtkDoubleArray()
persistenceArrayOut.SetName("Persistence")
timeArrayOut = vtkCore.vtkIntArray()
timeArrayOut.SetName("TimeStepIndex")

offsets = []  # global starting point index for each frame
idx = 0
for frameIdx, (coords, pers) in enumerate(maximaPerFrame):
    offsets.append(idx)
    for p, per in zip(coords, pers):
        points.InsertNextPoint(*p)
        persistenceArrayOut.InsertNextValue(per)
        timeArrayOut.InsertNextValue(frameIdx)
        idx += 1

lines = vtkDM.vtkCellArray()
for frameIdx, rowIdx, colIdx in trackEdges:
    line = vtkDM.vtkLine()
    line.GetPointIds().SetId(0, offsets[frameIdx] + rowIdx)
    line.GetPointIds().SetId(1, offsets[frameIdx + 1] + colIdx)
    lines.InsertNextCell(line)

tracksPolyData = vtkDM.vtkPolyData()
tracksPolyData.SetPoints(points)
tracksPolyData.SetLines(lines)
tracksPolyData.GetPointData().AddArray(persistenceArrayOut)
tracksPolyData.GetPointData().AddArray(timeArrayOut)

tracksMaxima = TrivialProducer(registrationName="TrackedMaxima")
tracksMaxima.GetClientSideObject().SetOutput(tracksPolyData)
tracksMaxima.UpdatePipeline()

# ---------------------------------------------------------------------------
# 5. Visualize the scalar field together with the tracked critical points
#    (spheres of radius 2, warm-cold colormap)
# ---------------------------------------------------------------------------
Show(reader, view)
reader.UpdatePipeline(trackTimes[-1])
readerDisp2 = GetDisplayProperties(reader, view)
ColorBy(readerDisp2, ("POINTS", SCALAR_ARRAY))
readerDisp2.SetScalarBarVisibility(view, True)
readerDisp2.LookupTable = GetColorTransferFunction(SCALAR_ARRAY)
readerDisp2.LookupTable.ApplyPreset("Cool to Warm", True)
readerDisp2.Opacity = 0.6

glyph = Glyph(Input=tracksMaxima, GlyphType="Sphere")
glyph.GlyphType.Radius = SPHERE_RADIUS
glyph.ScaleFactor = 1.0
glyph.GlyphMode = "All Points"

glyphDisp = Show(glyph, view)
ColorBy(glyphDisp, ("POINTS", "Persistence"))
glyphDisp.SetScalarBarVisibility(view, True)
persistenceLUT = GetColorTransferFunction("Persistence")
persistenceLUT.ApplyPreset("Cool to Warm", True)

trackDisp = Show(tracksMaxima, view)
trackDisp.SetRepresentationType("Wireframe")
trackDisp.LineWidth = 3.0
ColorBy(trackDisp, ("POINTS", "Persistence"))

view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "3_tracked_maxima_with_scalar_field.png"), view)

print("Done. Screenshots written to", OUT_DIR)
