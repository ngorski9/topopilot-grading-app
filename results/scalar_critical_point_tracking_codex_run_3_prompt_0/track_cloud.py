"""TTK persistence simplification and EMD tracking for cloud1..cloud3."""
import os, subprocess
import numpy as np
import ot
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

ROOT = "/workspace"
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)
steps = [1, 2, 3]
threshold = 0.5

def read_vti(path):
    r = vtk.vtkXMLImageDataReader(); r.SetFileName(path); r.Update()
    return r.GetOutput()

def diagram_for(step):
    prefix = os.path.join(OUT, "persistence_%d" % step)
    output = prefix + "_port_0.vtu"
    if not os.path.exists(output):
        subprocess.run(["/opt/conda/bin/ttkPersistenceDiagramCmd", "-i",
                        os.path.join(ROOT, "cloud", "cloud%d.vti" % step),
                        "-a", "Scalars_", "-o", prefix, "-d", "1"], check=True)
    r = vtk.vtkXMLUnstructuredGridReader(); r.SetFileName(output); r.Update()
    return r.GetOutput()

def simplified_maxima(step):
    """Keep max endpoints of TTK persistence pairs >= requested threshold."""
    d = diagram_for(step)
    pers = d.GetCellData().GetArray("Persistence")
    ctype = d.GetPointData().GetArray("CriticalType")
    value = d.GetPointData().GetArray("ttkVertexScalarField")
    candidates = {}
    for cid in range(d.GetNumberOfCells()):
        if pers.GetTuple1(cid) < threshold: continue
        cell = d.GetCell(cid)
        for j in range(cell.GetNumberOfPoints()):
            pid = cell.GetPointId(j)
            if int(ctype.GetTuple1(pid)) == 3: # TTK maximum
                candidates[pid] = (np.array(d.GetPoint(pid)), value.GetTuple1(pid), pers.GetTuple1(cid))
    # Persistence 0.5 retains all non-zero integer-valued features; keep this exact threshold.
    return list(candidates.values())

fields = [read_vti(os.path.join(ROOT, "cloud", "cloud%d.vti" % s)) for s in steps]
maxima = [simplified_maxima(s) for s in steps]

# Exact earth mover plans between uniform maxima measures.  Nonzero plan edges define tracks.
plans = []
for a, b in zip(maxima[:-1], maxima[1:]):
    xa = np.array([q[0] for q in a]); xb = np.array([q[0] for q in b])
    cost = ot.dist(xa, xb, metric="euclidean")
    plan = ot.emd(np.ones(len(a))/len(a), np.ones(len(b))/len(b), cost)
    plans.append(plan)

# Assign consistent track ids, retaining each positive EMD coupling as a PL segment.
track_ids = [np.full(len(maxima[0]), -1, dtype=int)]
track_ids[0][:] = np.arange(len(maxima[0]))
next_id = len(maxima[0])
for k, plan in enumerate(plans):
    ids = np.full(len(maxima[k+1]), -1, dtype=int)
    for j in range(plan.shape[1]):
        parents = np.where(plan[:, j] > 1e-12)[0]
        if len(parents): ids[j] = track_ids[k][parents[np.argmax(plan[parents, j])]]
        else: ids[j] = next_id; next_id += 1
    track_ids.append(ids)

def points_poly(points, ids, time):
    p = vtk.vtkPolyData(); pts = vtk.vtkPoints()
    scalars = vtk.vtkDoubleArray(); scalars.SetName("Scalar")
    tids = vtk.vtkIntArray(); tids.SetName("TrackId")
    for q, tid in zip(points, ids):
        pts.InsertNextPoint(*q[0]); scalars.InsertNextValue(q[1]); tids.InsertNextValue(int(tid))
    p.SetPoints(pts); p.GetPointData().AddArray(scalars); p.GetPointData().SetScalars(scalars); p.GetPointData().AddArray(tids)
    return p

# Build all tracked critical points and EMD PL correspondence segments for interoperable output.
append = vtk.vtkAppendPolyData()
for k in range(3): append.AddInputData(points_poly(maxima[k], track_ids[k], steps[k]))
append.Update()
w = vtk.vtkXMLPolyDataWriter(); w.SetFileName(os.path.join(OUT, "tracked_maxima.vtp")); w.SetInputData(append.GetOutput()); w.Write()

lines = vtk.vtkPolyData(); lp = vtk.vtkPoints(); lc = vtk.vtkCellArray(); weights = vtk.vtkDoubleArray(); weights.SetName("EMDWeight")
for k, plan in enumerate(plans):
    for i, j in zip(*np.where(plan > 1e-12)):
        ia = lp.InsertNextPoint(*maxima[k][i][0]); ib = lp.InsertNextPoint(*maxima[k+1][j][0])
        line = vtk.vtkLine(); line.GetPointIds().SetId(0, ia); line.GetPointIds().SetId(1, ib); lc.InsertNextCell(line); weights.InsertNextValue(plan[i,j])
lines.SetPoints(lp); lines.SetLines(lc); lines.GetCellData().SetScalars(weights)
lw = vtk.vtkXMLPolyDataWriter(); lw.SetFileName(os.path.join(OUT, "emd_tracks.vtp")); lw.SetInputData(lines); lw.Write()

# ParaView-compatible VTK rendering: original scalar fields, warm-cold map, radius-2 maxima.
renwin = vtk.vtkRenderWindow(); renwin.SetSize(1800, 600); renwin.SetOffScreenRendering(1)
lut = vtk.vtkColorTransferFunction(); lut.AddRGBPoint(0, 0.23,0.30,0.75); lut.AddRGBPoint(25, 0.95,0.95,0.95); lut.AddRGBPoint(50, 0.70,0.02,0.15)
for k, field in enumerate(fields):
    ren = vtk.vtkRenderer(); ren.SetViewport(k/3.0, 0, (k+1)/3.0, 1); ren.SetBackground(0.08,0.08,0.10); renwin.AddRenderer(ren)
    fm = vtk.vtkDataSetMapper(); fm.SetInputData(field); fm.SetScalarModeToUsePointFieldData(); fm.SelectColorArray("Scalars_"); fm.SetLookupTable(lut); fm.SetScalarRange(0,50)
    fa = vtk.vtkActor(); fa.SetMapper(fm); ren.AddActor(fa)
    glyph = vtk.vtkGlyph3D(); glyph.SetInputData(points_poly(maxima[k], track_ids[k], steps[k])); sphere = vtk.vtkSphereSource(); sphere.SetRadius(2); sphere.SetThetaResolution(16); sphere.SetPhiResolution(16); glyph.SetSourceConnection(sphere.GetOutputPort()); glyph.ScalingOff(); glyph.Update()
    gm = vtk.vtkPolyDataMapper(); gm.SetInputConnection(glyph.GetOutputPort()); gm.SetScalarModeToUsePointFieldData(); gm.SelectColorArray("Scalar"); gm.SetLookupTable(lut); gm.SetScalarRange(0,50)
    ga = vtk.vtkActor(); ga.SetMapper(gm); ga.GetProperty().SetSpecular(0.4); ren.AddActor(ga)
    text = vtk.vtkTextActor(); text.SetInput("Original scalar field + persistence >= 0.5 maxima | time %d" % steps[k]); text.GetTextProperty().SetFontSize(18); text.GetTextProperty().SetColor(1,1,1); text.SetPosition(12, 12); ren.AddActor2D(text)
    ren.GetActiveCamera().SetPosition(128,128,350); ren.GetActiveCamera().SetFocalPoint(128,128,0); ren.ResetCamera(); ren.GetActiveCamera().ParallelProjectionOn(); ren.ResetCamera()
renwin.Render(); grab = vtk.vtkWindowToImageFilter(); grab.SetInput(renwin); grab.Update(); png = vtk.vtkPNGWriter(); png.SetFileName(os.path.join(OUT, "cloud_tracking.png")); png.SetInputConnection(grab.GetOutputPort()); png.Write()

with open(os.path.join(OUT, "README.txt"), "w") as f:
    f.write("TTK persistence-diagram simplification threshold: 0.5\n")
    f.write("EMD tracked piecewise-linear maxima across cloud1.vti, cloud2.vti, cloud3.vti\n")
    f.write("maxima counts: %s\n" % [len(x) for x in maxima])
    f.write("sphere radius: 2; colormap: warm-cold\n")
print("maxima per time step:", [len(x) for x in maxima])
