import paraview.simple as ps
import numpy as np
from vtkmodules.util import numpy_support as ns
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkCommonCore import vtkPoints

N_STEPS = 3
Z_GAP = 180.0
files = [f"/workspace/cylinder/cylinder{i+1}.vti" for i in range(N_STEPS)]

readers = []
calcs = []
for i, fn in enumerate(files):
    r = ps.OpenDataFile(fn)
    ps.UpdatePipeline(time=0, proxy=r)
    # combine u,v into a vector array 'vec' (for glyphing) and magnitude scalar (for tracking)
    calc_vec = ps.Calculator(Input=r)
    calc_vec.ResultArrayName = "vec"
    calc_vec.AttributeType = "Point Data"
    calc_vec.Function = "u*iHat+v*jHat"

    calc_mag = ps.Calculator(Input=calc_vec)
    calc_mag.ResultArrayName = f"mag_t{i}"
    calc_mag.AttributeType = "Point Data"
    calc_mag.Function = "sqrt(u*u+v*v)"

    ps.UpdatePipeline(time=0, proxy=calc_mag)
    readers.append(r)
    calcs.append(calc_mag)

# Keep only the magnitude array (drop u,v,vec) on each branch, then append into
# one dataset with 3 point-data arrays (one per timestep, in order).
passers = []
for i, c in enumerate(calcs):
    p = ps.PassArrays(Input=c)
    p.PointDataArrays = [f"mag_t{i}"]
    ps.UpdatePipeline(time=0, proxy=p)
    passers.append(p)

merged = ps.AppendAttributes(Input=passers)
ps.UpdatePipeline(time=0, proxy=merged)

# --- Track critical points across the 3 timesteps using TTK's partial-OT
# persistence-diagram matcher (TrackingFromFields: sparse-Munkres Wasserstein
# matching lets a critical point be matched to the diagonal, i.e. born/killed,
# rather than forcing a full bijection between consecutive timesteps -- this
# is the "partial" optimal transport referenced in the task).
tracking = ps.TTKTrackingFromFields(Input=merged)
tracking.Firsttimestep = 0
tracking.Lasttimestep = -1
tracking.Timesampling = 1
tracking.Persistencethreshold = 0.15
tracking.pparameter = "2"
tracking.Extremumweight = 1.0
tracking.Saddleweight = 1.0
tracking.Xweight = 1.0
tracking.Yweight = 1.0
tracking.Zweight = 0.0
ps.UpdatePipeline(time=0, proxy=tracking)

trackOut = ps.servermanager.Fetch(tracking)
print("Tracking output type:", trackOut.GetClassName())
print("N points:", trackOut.GetNumberOfPoints(), "N cells:", trackOut.GetNumberOfCells())

# The raw output groups each tracked feature's points under a shared
# ConnectedComponentId (a full trajectory chain across all 3 input fields).
# Recover per-trajectory chronological order from point-append order (points
# belonging to the same physical location are emitted in temporal sequence as
# TTK sweeps consecutive pairs (t0,t1) then (t1,t2)), dedupe coincident
# locations, and lay each trajectory out along Z accordingly.
pts_all = ns.vtk_to_numpy(trackOut.GetPoints().GetData())[:, :2]
cc = ns.vtk_to_numpy(trackOut.GetPointData().GetArray("ConnectedComponentId"))

groups = {}
for idx in range(pts_all.shape[0]):
    groups.setdefault(int(cc[idx]), []).append(idx)

new_points = []
new_lines = []
new_cc = []
new_rank = []
for gid, idxs in groups.items():
    seen = {}
    ordered = []
    for idx in idxs:  # idxs already in append (chronological) order
        key = (round(float(pts_all[idx, 0]), 3), round(float(pts_all[idx, 1]), 3))
        if key not in seen:
            seen[key] = True
            ordered.append(key)
    ordered = ordered[:N_STEPS]
    base = len(new_points)
    for rank, (x, y) in enumerate(ordered):
        new_points.append([x, y, rank * Z_GAP])
        new_cc.append(gid)
        new_rank.append(rank)
    for rank in range(len(ordered) - 1):
        new_lines.append([base + rank, base + rank + 1])

new_points = np.array(new_points, dtype=float)
new_cc = np.array(new_cc, dtype=np.int32)
new_rank = np.array(new_rank, dtype=np.int32)

print("Trajectories:", len(groups))
print("Rank (recovered global timestep) histogram:",
      {int(t): int((new_rank == t).sum()) for t in sorted(set(new_rank.tolist()))})

grid = vtkUnstructuredGrid()
vpts = vtkPoints()
vpts.SetData(ns.numpy_to_vtk(new_points, deep=True))
grid.SetPoints(vpts)
grid.Allocate(len(new_lines))
for ids in new_lines:
    grid.InsertNextCell(3, len(ids), ids)  # VTK_LINE = 3

cc_arr = ns.numpy_to_vtk(new_cc, deep=True)
cc_arr.SetName("TrajectoryId")
grid.GetPointData().AddArray(cc_arr)

rank_arr = ns.numpy_to_vtk(new_rank, deep=True)
rank_arr.SetName("Timestep")
grid.GetPointData().AddArray(rank_arr)

tracked = ps.TrivialProducer()
tracked.GetClientSideObject().SetOutput(grid)
ps.UpdatePipeline(time=0, proxy=tracked)

# ---------------- Visualization ----------------
view = ps.CreateRenderView()
view.ViewSize = [1600, 1200]
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]
view.UseColorPaletteForBackground = 0

# Vector field glyphs for each timestep, offset in Z to align with the
# tracked critical points living at the same timestep.
colors = [[0.2, 0.4, 0.9], [0.2, 0.8, 0.3], [0.9, 0.3, 0.2]]
for i, c in enumerate(calcs):
    transform = ps.Transform(Input=c)
    transform.Transform = "Transform"
    transform.Transform.Translate = [0.0, 0.0, i * Z_GAP]
    ps.UpdatePipeline(time=0, proxy=transform)

    glyph = ps.Glyph(Input=transform, GlyphType="Arrow")
    glyph.OrientationArray = ["POINTS", "vec"]
    glyph.ScaleArray = ["POINTS", "vec"]
    glyph.ScaleFactor = 70.0
    glyph.GlyphMode = "Every Nth Point"
    glyph.Stride = 211
    ps.UpdatePipeline(time=0, proxy=glyph)

    disp = ps.Show(glyph, view)
    disp.Representation = "Surface"
    disp.ColorArrayName = ["POINTS", ""]
    disp.AmbientColor = colors[i]
    disp.DiffuseColor = colors[i]
    disp.Opacity = 0.85

tdisp = ps.Show(tracked, view)
tdisp.Representation = "Surface"
tdisp.LineWidth = 3.0
ps.ColorBy(tdisp, ("POINTS", "TrajectoryId"))
tdisp.RescaleTransferFunctionToDataRange(True)
lut = ps.GetColorTransferFunction("TrajectoryId")
lut.ApplyPreset("Rainbow Desaturated", True)

pdisp = ps.Show(tracked, view)
pdisp.Representation = "Points"
pdisp.PointSize = 9.0
pdisp.RenderPointsAsSpheres = 1
ps.ColorBy(pdisp, ("POINTS", "TrajectoryId"))

ps.ResetCamera(view)
cam = view.GetActiveCamera()
cam.Azimuth(25)
cam.Elevation(15)
ps.ResetCamera(view)

ps.Render(view)
ps.SaveScreenshot("/workspace/cylinder_tracked_critical_points.png", view, ImageResolution=[1600, 1200])
print("Saved /workspace/cylinder_tracked_critical_points.png")


ps.SaveData("/workspace/cylinder_tracking.vtu", proxy=tracked)
print("Saved /workspace/cylinder_tracking.vtu")
