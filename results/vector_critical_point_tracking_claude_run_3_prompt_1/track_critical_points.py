"""
Track critical points of a time-dependent 2D piecewise-linear vector field
(cylinder dataset, arrays 'u' and 'v') across 3 time steps using partial
optimal transport (POT), then render the tracked points together with the
original vector field.
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import ot

N_STEPS = 3
FILES = [f"./cylinder/cylinder{i}.vti" for i in range(1, N_STEPS + 1)]


def read_field(path):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(path)
    r.Update()
    img = r.GetOutput()
    dims = img.GetDimensions()
    u = vtk_to_numpy(img.GetPointData().GetArray("u")).reshape(dims[1], dims[0])
    v = vtk_to_numpy(img.GetPointData().GetArray("v")).reshape(dims[1], dims[0])
    return img, u, v, dims


def find_critical_points(img, u, v, dims):
    """Find zeros of the PL-interpolated (u,v) field on a triangulated
    version of the pixel grid (2 triangles per quad cell). Classify each
    by the Poincare index sign (source/sink vs saddle)."""
    nx, ny = dims[0], dims[1]
    origin = img.GetOrigin()
    spacing = img.GetSpacing()

    def phys(i, j):
        return np.array([origin[0] + i * spacing[0], origin[1] + j * spacing[1]])

    pts = []
    types = []

    def tri_zero(pA, vA, pB, vB, pC, vC):
        # Solve for barycentric (l1,l2,l3) s.t. l1*vA+l2*vB+l3*vC = 0, sum=1
        M = np.array([[vA[0], vB[0], vC[0]],
                      [vA[1], vB[1], vC[1]],
                      [1.0, 1.0, 1.0]])
        rhs = np.array([0.0, 0.0, 1.0])
        try:
            lam = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return None
        if np.all(lam >= -1e-9) and np.all(lam <= 1 + 1e-9):
            p = lam[0] * pA + lam[1] * pB + lam[2] * pC
            # Poincare index: sign of the signed area (orientation) mapped
            # by the vector field relative to geometric orientation.
            def cross2(a, b):
                return a[0] * b[1] - a[1] * b[0]
            geom_area = cross2(pB - pA, pC - pA)
            vec_area = cross2(vB - vA, vC - vA)
            index = np.sign(vec_area * geom_area) if geom_area != 0 else 0
            return p, index
        return None

    for j in range(ny - 1):
        for i in range(nx - 1):
            p00, p10, p01, p11 = phys(i, j), phys(i + 1, j), phys(i, j + 1), phys(i + 1, j + 1)
            v00 = np.array([u[j, i], v[j, i]])
            v10 = np.array([u[j, i + 1], v[j, i + 1]])
            v01 = np.array([u[j + 1, i], v[j + 1, i]])
            v11 = np.array([u[j + 1, i + 1], v[j + 1, i + 1]])
            for (pA, vA, pB, vB, pC, vC) in (
                (p00, v00, p10, v10, p11, v11),
                (p00, v00, p11, v11, p01, v01),
            ):
                res = tri_zero(pA, vA, pB, vB, pC, vC)
                if res is not None:
                    p, idx = res
                    pts.append(p)
                    types.append(idx)

    pts = np.array(pts) if pts else np.zeros((0, 2))
    types = np.array(types) if types else np.zeros((0,))
    # De-duplicate points that are extremely close (shared triangle edges).
    if len(pts) > 1:
        keep = np.ones(len(pts), dtype=bool)
        for a in range(len(pts)):
            if not keep[a]:
                continue
            for b in range(a + 1, len(pts)):
                if keep[b] and np.linalg.norm(pts[a] - pts[b]) < 1e-6:
                    keep[b] = False
        pts, types = pts[keep], types[keep]
    return pts, types


def partial_ot_match(P0, P1, keep_frac=0.85):
    """Match critical points between two consecutive time steps using
    partial optimal transport: only a fraction of the mass is required
    to be transported, so points that appear/disappear are naturally
    left unmatched (partial mass, à la Lacombe et al. tracking)."""
    n0, n1 = len(P0), len(P1)
    if n0 == 0 or n1 == 0:
        return []
    a = np.ones(n0) / n0
    b = np.ones(n1) / n1
    C = ot.dist(P0, P1, metric="euclidean")
    m = keep_frac * min(a.sum(), b.sum())
    G = ot.partial.partial_wasserstein(a, b, C, m=m)
    matches = []
    for i in range(n0):
        j = np.argmax(G[i])
        if G[i, j] > 1e-12:
            matches.append((i, j, G[i, j]))
    return matches


# ---------------------------------------------------------------------
# 1. Extract critical points per time step
# ---------------------------------------------------------------------
imgs, fields, crit_pts, crit_types = [], [], [], []
for f in FILES:
    img, u, v, dims = read_field(f)
    pts, types = find_critical_points(img, u, v, dims)
    imgs.append(img)
    fields.append((u, v, dims))
    crit_pts.append(pts)
    crit_types.append(types)
    print(f"{f}: {len(pts)} critical points")

# ---------------------------------------------------------------------
# 2. Track critical points across the 3 steps via partial optimal transport
# ---------------------------------------------------------------------
tracks = []  # list of dict: {step: point_index}
active = {i: [(0, i)] for i in range(len(crit_pts[0]))}  # track_id -> path so far
next_id = len(crit_pts[0])
current = dict(active)

for t in range(N_STEPS - 1):
    matches = partial_ot_match(crit_pts[t], crit_pts[t + 1])
    matched_src = {i for i, j, w in matches}
    matched_dst = {j for i, j, w in matches}
    new_current = {}
    # continue matched tracks
    src_to_trackid = {}
    for tid, path in current.items():
        last_t, last_i = path[-1]
        if last_t == t:
            src_to_trackid[last_i] = tid
    for i, j, w in matches:
        tid = src_to_trackid.get(i)
        if tid is None:
            continue
        path = current[tid] + [(t + 1, j)]
        new_current[tid] = path
    # keep unmatched existing tracks alive but frozen (they died out)
    for tid, path in current.items():
        if tid not in new_current:
            tracks.append(path)
    # start new tracks for unmatched destination points (newly appeared)
    for j in range(len(crit_pts[t + 1])):
        if j not in matched_dst:
            new_current[next_id] = [(t + 1, j)]
            next_id += 1
    current = new_current

for tid, path in current.items():
    tracks.append(path)

print(f"{len(tracks)} tracks found across {N_STEPS} time steps")
for path in tracks:
    print("  " + " -> ".join(f"t{t}:p{i}" for t, i in path))

# ---------------------------------------------------------------------
# 3. Build VTK objects for rendering: glyphed vector fields (one per step,
#    offset in Z for visual separation) + critical points + track lines.
# ---------------------------------------------------------------------
renderer = vtk.vtkRenderer()
renderer.SetBackground(1, 1, 1)

Z_OFFSET = 130.0  # separate the 3 time steps visually along Z

for t, img in enumerate(imgs):
    u, v, dims = fields[t]
    # subsample for glyphing so arrows are visible
    stride = 6
    pts = vtk.vtkPoints()
    vecs = vtk.vtkDoubleArray()
    vecs.SetNumberOfComponents(3)
    vecs.SetName("vec")
    mags = vtk.vtkDoubleArray()
    mags.SetName("mag")
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    nx, ny = dims[0], dims[1]
    for j in range(0, ny, stride):
        for i in range(0, nx, stride):
            x = origin[0] + i * spacing[0]
            y = origin[1] + j * spacing[1]
            z = t * Z_OFFSET
            pts.InsertNextPoint(x, y, z)
            uu, vv = u[j, i], v[j, i]
            vecs.InsertNextTuple3(uu, vv, 0.0)
            mags.InsertNextValue(float(np.hypot(uu, vv)))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.GetPointData().SetVectors(vecs)
    pd.GetPointData().SetScalars(mags)

    arrow = vtk.vtkArrowSource()
    glyph = vtk.vtkGlyph3D()
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetInputData(pd)
    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByVector()
    glyph.SetScaleFactor(45.0)
    glyph.OrientOn()
    glyph.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("mag")
    mapper.SetScalarRange(0, 1.0)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.55)
    renderer.AddActor(actor)

    # faint plane behind each field to mark the time-step slice
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(origin[0], origin[1], t * Z_OFFSET)
    plane.SetPoint1(origin[0] + nx * spacing[0], origin[1], t * Z_OFFSET)
    plane.SetPoint2(origin[0], origin[1] + ny * spacing[1], t * Z_OFFSET)
    pmapper = vtk.vtkPolyDataMapper()
    pmapper.SetInputConnection(plane.GetOutputPort())
    pactor = vtk.vtkActor()
    pactor.SetMapper(pmapper)
    pactor.GetProperty().SetColor(0.9, 0.9, 0.9)
    pactor.GetProperty().SetOpacity(0.15)
    renderer.AddActor(pactor)

# critical points, colored by type (saddle vs source/sink)
cp_points = vtk.vtkPoints()
cp_colors = vtk.vtkUnsignedCharArray()
cp_colors.SetNumberOfComponents(3)
cp_colors.SetName("Colors")
cp_verts = vtk.vtkCellArray()
pid = 0
point_index_map = {}  # (t, i) -> global point id in cp_points
for t in range(N_STEPS):
    for i, (p, ty) in enumerate(zip(crit_pts[t], crit_types[t])):
        z = t * Z_OFFSET
        cp_points.InsertNextPoint(p[0], p[1], z)
        cp_verts.InsertNextCell(1)
        cp_verts.InsertCellPoint(pid)
        if ty > 0:
            cp_colors.InsertNextTuple3(220, 30, 30)   # source/sink -> red
        elif ty < 0:
            cp_colors.InsertNextTuple3(30, 60, 220)   # saddle -> blue
        else:
            cp_colors.InsertNextTuple3(120, 120, 120)
        point_index_map[(t, i)] = pid
        pid += 1

cp_pd = vtk.vtkPolyData()
cp_pd.SetPoints(cp_points)
cp_pd.SetVerts(cp_verts)
cp_pd.GetPointData().SetScalars(cp_colors)

sphere = vtk.vtkSphereSource()
sphere.SetRadius(4.5)
cp_glyph = vtk.vtkGlyph3D()
cp_glyph.SetSourceConnection(sphere.GetOutputPort())
cp_glyph.SetInputData(cp_pd)
cp_glyph.SetColorModeToColorByScalar()
cp_glyph.ScalingOff()
cp_glyph.Update()

cp_mapper = vtk.vtkPolyDataMapper()
cp_mapper.SetInputConnection(cp_glyph.GetOutputPort())
cp_actor = vtk.vtkActor()
cp_actor.SetMapper(cp_mapper)
renderer.AddActor(cp_actor)

# track lines (partial-OT correspondences across time)
line_pts = vtk.vtkPoints()
line_cells = vtk.vtkCellArray()
lp = 0
for path in tracks:
    if len(path) < 2:
        continue
    ids = []
    for (t, i) in path:
        p = crit_pts[t][i]
        line_pts.InsertNextPoint(p[0], p[1], t * Z_OFFSET)
        ids.append(lp)
        lp += 1
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(ids))
    for k, pid_ in enumerate(ids):
        line.GetPointIds().SetId(k, pid_)
    line_cells.InsertNextCell(line)

line_pd = vtk.vtkPolyData()
line_pd.SetPoints(line_pts)
line_pd.SetLines(line_cells)
tube = vtk.vtkTubeFilter()
tube.SetInputData(line_pd)
tube.SetRadius(1.0)
tube.SetNumberOfSides(8)
tube_mapper = vtk.vtkPolyDataMapper()
tube_mapper.SetInputConnection(tube.GetOutputPort())
tube_actor = vtk.vtkActor()
tube_actor.SetMapper(tube_mapper)
tube_actor.GetProperty().SetColor(0.05, 0.6, 0.05)
renderer.AddActor(tube_actor)

renderer.ResetCamera()
cam = renderer.GetActiveCamera()
cam.Azimuth(35)
cam.Elevation(25)
cam.Zoom(1.3)
renderer.ResetCameraClippingRange()

render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(1)
render_window.AddRenderer(renderer)
render_window.SetSize(1400, 900)
render_window.Render()

w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(render_window)
w2i.Update()
writer = vtk.vtkPNGWriter()
writer.SetFileName("/workspace/cylinder_tracked_critical_points.png")
writer.SetInputConnection(w2i.GetOutputPort())
writer.Write()
print("Wrote /workspace/cylinder_tracked_critical_points.png")

# Also dump the polydata (critical points + tracks) for inspection in ParaView
appender = vtk.vtkAppendPolyData()
appender.AddInputData(cp_glyph.GetOutput())
appender.Update()
w = vtk.vtkXMLPolyDataWriter()
w.SetFileName("/workspace/cylinder_critical_points.vtp")
w.SetInputConnection(appender.GetOutputPort())
w.Write()

w2 = vtk.vtkXMLPolyDataWriter()
w2.SetFileName("/workspace/cylinder_tracks.vtp")
w2.SetInputData(line_pd)
w2.Write()
print("Wrote cylinder_critical_points.vtp and cylinder_tracks.vtp")
