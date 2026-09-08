"""
Track critical points of a time-dependent piecewise-linear vector field
(TTK 'cylinder' dataset, arrays 'u','v') across 3 time steps using
partial optimal transport (POT), and render the result together with
the original vector field using ParaView.

Run with: pvpython track_critical_points.py
"""
import os
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import ot

DATA_DIR = "./cylinder"
TIME_FILES = ["cylinder1.vti", "cylinder2.vti", "cylinder3.vti"]  # 3 time steps
OUT_DIR = "./cylinder_tracking_out"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Load each timestep, build (u, v) vector field on the point grid
# ----------------------------------------------------------------------
def load_field(path):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(path)
    r.Update()
    img = r.GetOutput()
    dims = img.GetDimensions()          # (nx, ny, 1)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    pd = img.GetPointData()
    u = vtk_to_numpy(pd.GetArray("u")).reshape(dims[1], dims[0])
    v = vtk_to_numpy(pd.GetArray("v")).reshape(dims[1], dims[0])
    return img, dims, spacing, origin, u, v


# ----------------------------------------------------------------------
# 2. Extract critical points of the PL vector field.
#    Each grid cell (quad) is split into 2 triangles; on each triangle
#    (u,v) is linear in barycentric coordinates, so we solve for the
#    zero of the linear map and keep it if it lies inside the triangle.
#    The Poincare index (sign of the local Jacobian determinant) is
#    used to classify the critical point.
# ----------------------------------------------------------------------
def critical_points(dims, spacing, origin, u, v):
    nx, ny = dims[0], dims[1]
    ox, oy = origin[0], origin[1]
    sx, sy = spacing[0], spacing[1]
    pts = []
    types = []

    def solve_tri(p0, p1, p2, f0, f1, f2):
        # f_k = (u_k, v_k). Find barycentric (l1, l2) s.t.
        # f0 + l1*(f1-f0) + l2*(f2-f0) = 0, l1,l2>=0, l1+l2<=1
        A = np.array([[f1[0] - f0[0], f2[0] - f0[0]],
                      [f1[1] - f0[1], f2[1] - f0[1]]])
        b = -np.array(f0)
        det = np.linalg.det(A)
        if abs(det) < 1e-14:
            return None
        l = np.linalg.solve(A, b)
        l1, l2 = l
        if l1 < -1e-9 or l2 < -1e-9 or (l1 + l2) > 1 + 1e-9:
            return None
        l0 = 1 - l1 - l2
        pos = l0 * np.array(p0) + l1 * np.array(p1) + l2 * np.array(p2)
        index = 1 if det > 0 else -1
        return pos, index

    for j in range(ny - 1):
        for i in range(nx - 1):
            # corners of cell (i,j)
            p00 = (ox + i * sx, oy + j * sy)
            p10 = (ox + (i + 1) * sx, oy + j * sy)
            p01 = (ox + i * sx, oy + (j + 1) * sy)
            p11 = (ox + (i + 1) * sx, oy + (j + 1) * sy)
            f00 = (u[j, i], v[j, i])
            f10 = (u[j, i + 1], v[j, i + 1])
            f01 = (u[j + 1, i], v[j + 1, i])
            f11 = (u[j + 1, i + 1], v[j + 1, i + 1])

            for (a, b, c, fa, fb, fc) in (
                (p00, p10, p11, f00, f10, f11),
                (p00, p11, p01, f00, f11, f01),
            ):
                res = solve_tri(a, b, c, fa, fb, fc)
                if res is not None:
                    pos, index = res
                    pts.append(pos)
                    types.append(index)

    return np.array(pts).reshape(-1, 2) if pts else np.zeros((0, 2)), \
        np.array(types, dtype=int)


frames = []  # list of dict(img, dims, spacing, origin, u, v, pts, types)
for f in TIME_FILES:
    img, dims, spacing, origin, u, v = load_field(os.path.join(DATA_DIR, f))
    pts, types = critical_points(dims, spacing, origin, u, v)
    frames.append(dict(img=img, dims=dims, spacing=spacing, origin=origin,
                        u=u, v=v, pts=pts, types=types))
    print(f"{f}: {len(pts)} critical points "
          f"({(types == 1).sum()} sources/sinks-like, {(types == -1).sum()} saddle-like)")


# ----------------------------------------------------------------------
# 3. Track critical points frame-to-frame with PARTIAL optimal transport.
#    Each critical point carries unit mass. Because the number of
#    critical points can change between frames (creation/destruction),
#    we transport only the mass that CAN be matched (m = min total mass)
#    using ot.partial.partial_wasserstein, with cost = Euclidean distance
#    (critical points of different type get an infinite/very large cost
#    so they cannot be matched to each other).
# ----------------------------------------------------------------------
def partial_ot_match(pts_a, types_a, pts_b, types_b, max_dist=40.0):
    na, nb = len(pts_a), len(pts_b)
    if na == 0 or nb == 0:
        return []
    C = np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1)
    BIG = 1e6
    mismatch = types_a[:, None] != types_b[None, :]
    C = C + mismatch * BIG

    a = np.ones(na) / na
    b = np.ones(nb) / nb
    m = min(a.sum(), b.sum())
    # discard pairs that are geometrically implausible before solving
    C_capped = np.minimum(C, BIG)
    gamma = ot.partial.partial_wasserstein(a, b, C_capped, m=m)

    matches = []
    thresh = gamma.max() * 1e-3 if gamma.max() > 0 else 0
    for i in range(na):
        for j in range(nb):
            if gamma[i, j] > max(thresh, 1e-12) and C[i, j] < BIG and C[i, j] < max_dist:
                matches.append((i, j, C[i, j]))
    return matches


matches_01 = partial_ot_match(frames[0]["pts"], frames[0]["types"],
                               frames[1]["pts"], frames[1]["types"])
matches_12 = partial_ot_match(frames[1]["pts"], frames[1]["types"],
                               frames[2]["pts"], frames[2]["types"])
print(f"matches t0->t1: {len(matches_01)}   matches t1->t2: {len(matches_12)}")


# ----------------------------------------------------------------------
# 4. Build a VTK polydata: all critical points (stacked with a Z offset
#    per time step so the 3 frames are visually separated) plus
#    polylines connecting matched points across time = the tracking graph.
# ----------------------------------------------------------------------
Z_STEP = 60.0  # separate the 3 time layers along Z for visualization

points = vtk.vtkPoints()
type_arr = []
time_arr = []
offsets = [0]
for t, fr in enumerate(frames):
    for (x, y), ty in zip(fr["pts"], fr["types"]):
        points.InsertNextPoint(x, y, t * Z_STEP)
        type_arr.append(ty)
        time_arr.append(t)
    offsets.append(points.GetNumberOfPoints())

poly = vtk.vtkPolyData()
poly.SetPoints(points)

verts = vtk.vtkCellArray()
for pid in range(points.GetNumberOfPoints()):
    verts.InsertNextCell(1, [pid])
poly.SetVerts(verts)

lines = vtk.vtkCellArray()
track_id_cell = []
next_track_id = 0
track_id_point = [-1] * points.GetNumberOfPoints()

for (i, j, d) in matches_01:
    pid_a = offsets[0] + i
    pid_b = offsets[1] + j
    lines.InsertNextCell(2, [pid_a, pid_b])
    tid = track_id_point[pid_a] if track_id_point[pid_a] != -1 else next_track_id
    if track_id_point[pid_a] == -1:
        next_track_id += 1
    track_id_point[pid_a] = tid
    track_id_point[pid_b] = tid
    track_id_cell.append(tid)

for (i, j, d) in matches_12:
    pid_a = offsets[1] + i
    pid_b = offsets[2] + j
    lines.InsertNextCell(2, [pid_a, pid_b])
    tid = track_id_point[pid_a] if track_id_point[pid_a] != -1 else next_track_id
    if track_id_point[pid_a] == -1:
        next_track_id += 1
    track_id_point[pid_a] = tid
    track_id_point[pid_b] = tid
    track_id_cell.append(tid)

poly.SetLines(lines)

for pid in range(points.GetNumberOfPoints()):
    if track_id_point[pid] == -1:
        track_id_point[pid] = next_track_id
        next_track_id += 1

vtk_type = numpy_to_vtk(np.array(type_arr, dtype=np.int32))
vtk_type.SetName("CriticalType")
poly.GetPointData().AddArray(vtk_type)

vtk_time = numpy_to_vtk(np.array(time_arr, dtype=np.int32))
vtk_time.SetName("TimeStep")
poly.GetPointData().AddArray(vtk_time)

vtk_track = numpy_to_vtk(np.array(track_id_point, dtype=np.int32))
vtk_track.SetName("TrackId")
poly.GetPointData().AddArray(vtk_track)

writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName(os.path.join(OUT_DIR, "tracked_critical_points.vtp"))
writer.SetInputData(poly)
writer.Write()
print("wrote", os.path.join(OUT_DIR, "tracked_critical_points.vtp"))


# ----------------------------------------------------------------------
# 5. Also stack the original vector fields (3 time steps, Z-offset the
#    same way) as vtkImageData -> vtkPolyData (points+vectors) so it can
#    be displayed as glyphs alongside the tracked critical points.
# ----------------------------------------------------------------------
vf_points = vtk.vtkPoints()
vf_vectors = []
vf_time = []
for t, fr in enumerate(frames):
    dims = fr["dims"]
    ox, oy = fr["origin"][0], fr["origin"][1]
    sx, sy = fr["spacing"][0], fr["spacing"][1]
    u, v = fr["u"], fr["v"]
    # subsample the grid for a readable glyph density
    step = 6
    for j in range(0, dims[1], step):
        for i in range(0, dims[0], step):
            vf_points.InsertNextPoint(ox + i * sx, oy + j * sy, t * Z_STEP)
            vf_vectors.append((u[j, i], v[j, i], 0.0))
            vf_time.append(t)

vf_poly = vtk.vtkPolyData()
vf_poly.SetPoints(vf_points)
vf_vec_arr = numpy_to_vtk(np.array(vf_vectors, dtype=np.float64))
vf_vec_arr.SetNumberOfComponents(3)
vf_vec_arr.SetName("vector")
vf_poly.GetPointData().SetVectors(vf_vec_arr)
vf_time_arr = numpy_to_vtk(np.array(vf_time, dtype=np.int32))
vf_time_arr.SetName("TimeStep")
vf_poly.GetPointData().AddArray(vf_time_arr)

vf_writer = vtk.vtkXMLPolyDataWriter()
vf_writer.SetFileName(os.path.join(OUT_DIR, "vector_field_glyph_points.vtp"))
vf_writer.SetInputData(vf_poly)
vf_writer.Write()
print("wrote", os.path.join(OUT_DIR, "vector_field_glyph_points.vtp"))


# ----------------------------------------------------------------------
# 6. Render everything with ParaView (off-screen) and save a PNG.
# ----------------------------------------------------------------------
from paraview.simple import (
    XMLPolyDataReader, Glyph, Show, Render, GetActiveViewOrCreate,
    ColorBy, GetColorTransferFunction, Tube, GetLayout, SaveScreenshot,
    ResetCamera, Text
)

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1400, 1000]
view.OrientationAxesVisibility = 0
view.Background = [1, 1, 1]

# vector field as arrow glyphs, colored by time step
vf_reader = XMLPolyDataReader(FileName=[os.path.join(OUT_DIR, "vector_field_glyph_points.vtp")])
glyph = Glyph(Input=vf_reader, GlyphType='Arrow')
glyph.OrientationArray = ['POINTS', 'vector']
glyph.ScaleArray = ['POINTS', 'vector']
glyph.ScaleFactor = 25.0
glyph.GlyphMode = 'All Points'
glyph_disp = Show(glyph, view)
ColorBy(glyph_disp, ('POINTS', 'TimeStep'))
glyph_disp.SetScalarBarVisibility(view, False)
glyph_ctf = GetColorTransferFunction('TimeStep')
glyph_ctf.RGBPoints = [0.0, 0.75, 0.75, 0.75,
                        1.0, 0.5, 0.5, 0.5,
                        2.0, 0.25, 0.25, 0.25]
glyph_disp.Opacity = 0.85

# tracking graph: critical points + tracking lines
cp_reader = XMLPolyDataReader(FileName=[os.path.join(OUT_DIR, "tracked_critical_points.vtp")])

tube = Tube(Input=cp_reader)
tube.Radius = 1.2
tube_disp = Show(tube, view)
ColorBy(tube_disp, ('POINTS', 'TrackId'))
tube_disp.SetScalarBarVisibility(view, False)

cp_disp = Show(cp_reader, view)
cp_disp.SetRepresentationType('Points')
cp_disp.PointSize = 14
ColorBy(cp_disp, ('POINTS', 'CriticalType'))
ctf = GetColorTransferFunction('CriticalType')
ctf.RGBPoints = [-1.0, 0.85, 0.1, 0.1,   # saddle-like -> red
                 1.0, 0.1, 0.3, 0.9]     # source/sink-like -> blue
cp_disp.SetScalarBarVisibility(view, True)

ResetCamera(view)
Render(view)
SaveScreenshot(os.path.join(OUT_DIR, "tracked_critical_points.png"), view,
                ImageResolution=[1400, 1000])
print("wrote", os.path.join(OUT_DIR, "tracked_critical_points.png"))
