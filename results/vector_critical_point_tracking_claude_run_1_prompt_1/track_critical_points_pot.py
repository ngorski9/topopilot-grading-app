"""
Track critical points of a time-dependent 2D piecewise-linear vector field
(cylinder dataset, arrays 'u','v') across 3 time steps using Partial Optimal
Transport (POT library), then render the tracks together with the original
vector field using VTK.
"""
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import ot
import ot.partial

DATA_DIR = "/workspace/cylinder"
TIMESTEPS = [1, 8, 16]          # three time steps to compare
Z_STEP = 40.0                   # vertical spacing used to separate time layers in the display
GLYPH_STRIDE = 8                # subsample factor for vector-field glyphs


# ---------------------------------------------------------------------------
# 1. I/O
# ---------------------------------------------------------------------------
def load_timestep(idx):
    fname = f"{DATA_DIR}/cylinder{idx}.vti"
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fname)
    reader.Update()
    img = reader.GetOutput()
    dims = img.GetDimensions()
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    pd = img.GetPointData()
    u = vtk_to_numpy(pd.GetArray("u")).reshape(dims[1], dims[0])
    v = vtk_to_numpy(pd.GetArray("v")).reshape(dims[1], dims[0])
    return img, dims, spacing, origin, u, v


# ---------------------------------------------------------------------------
# 2. Critical point extraction on the piecewise-linear (triangulated) field
#    Standard Poincare index / marching-triangles method.
# ---------------------------------------------------------------------------
def signed_angle(v0, v1):
    a0 = np.arctan2(v0[1], v0[0])
    a1 = np.arctan2(v1[1], v1[0])
    d = a1 - a0
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


def triangle_critical_point(pa, pb, pc, va, vb, vc):
    """Return (found, x, y, index, kind) for a triangle whose vector-field
    values at its 3 vertices are va, vb, vc (each length-2)."""
    turn = signed_angle(va, vb) + signed_angle(vb, vc) + signed_angle(vc, va)
    idx = int(round(turn / (2 * np.pi)))
    if idx == 0:
        return None

    # Solve barycentric coords l0*va + l1*vb + l2*vc = 0, l0+l1+l2 = 1
    A = np.array([[va[0], vb[0], vc[0]],
                  [va[1], vb[1], vc[1]],
                  [1.0, 1.0, 1.0]])
    rhs = np.array([0.0, 0.0, 1.0])
    try:
        lam = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None
    if lam.min() < -1e-6 or lam.max() > 1 + 1e-6:
        return None  # zero not actually inside triangle (degenerate/robustness)

    x = lam[0] * pa[0] + lam[1] * pb[0] + lam[2] * pc[0]
    y = lam[0] * pa[1] + lam[1] * pb[1] + lam[2] * pc[1]

    # Jacobian is constant over the (linear) triangle -> classify type
    M = np.array([[pb[0] - pa[0], pc[0] - pa[0]],
                  [pb[1] - pa[1], pc[1] - pa[1]]])
    Duv = np.array([[vb[0] - va[0], vc[0] - va[0]],
                    [vb[1] - va[1], vc[1] - va[1]]])
    J = Duv @ np.linalg.inv(M)
    det = np.linalg.det(J)
    trace = np.trace(J)

    if det < 0:
        kind = "saddle"
    else:
        disc = trace * trace - 4 * det
        if trace < 0:
            kind = "sink" if disc >= 0 else "attracting focus"
        else:
            kind = "source" if disc >= 0 else "repelling focus"
    return x, y, idx, kind


def extract_critical_points(dims, spacing, origin, u, v):
    nx, ny = dims[0], dims[1]
    xs = origin[0] + np.arange(nx) * spacing[0]
    ys = origin[1] + np.arange(ny) * spacing[1]

    pts = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            p00 = (xs[i], ys[j]); p10 = (xs[i + 1], ys[j])
            p01 = (xs[i], ys[j + 1]); p11 = (xs[i + 1], ys[j + 1])
            v00 = (u[j, i], v[j, i]); v10 = (u[j, i + 1], v[j, i + 1])
            v01 = (u[j + 1, i], v[j + 1, i]); v11 = (u[j + 1, i + 1], v[j + 1, i + 1])

            for (pa, pb, pc, va, vb, vc) in (
                (p00, p10, p01, v00, v10, v01),
                (p10, p11, p01, v10, v11, v01),
            ):
                res = triangle_critical_point(pa, pb, pc, va, vb, vc)
                if res is not None:
                    x, y, idx, kind = res
                    pts.append((x, y, idx, kind))
    return pts


# ---------------------------------------------------------------------------
# 3. Partial Optimal Transport matching between consecutive time steps
# ---------------------------------------------------------------------------
KIND_PENALTY = 25.0  # extra cost for matching critical points of different type


def match_partial_ot(pts_a, pts_b):
    na, nb = len(pts_a), len(pts_b)
    if na == 0 or nb == 0:
        return []
    A = np.array([[p[0], p[1]] for p in pts_a])
    B = np.array([[p[0], p[1]] for p in pts_b])
    M = ot.dist(A, B, metric="euclidean")
    for i in range(na):
        for j in range(nb):
            if pts_a[i][3] != pts_b[j][3]:
                M[i, j] += KIND_PENALTY

    a = np.ones(na) / na
    b = np.ones(nb) / nb
    m = 0.9 * min(a.sum(), b.sum())  # transport 90% of mass, rest = births/deaths

    gamma = ot.partial.partial_wasserstein(a, b, M, m=m)

    matches = []
    thresh = 0.5 * gamma.max() if gamma.max() > 0 else 0
    for i in range(na):
        j = np.argmax(gamma[i])
        if gamma[i, j] > 1e-8 and gamma[i, j] >= thresh * 0.2:
            matches.append((i, j))
    return matches


# ---------------------------------------------------------------------------
# 4. Build everything
# ---------------------------------------------------------------------------
def main():
    frames = []
    for t_idx, ts in enumerate(TIMESTEPS):
        img, dims, spacing, origin, u, v = load_timestep(ts)
        cps = extract_critical_points(dims, spacing, origin, u, v)
        print(f"time step {ts}: {len(cps)} critical points")
        frames.append(dict(ts=ts, t_idx=t_idx, img=img, dims=dims,
                            spacing=spacing, origin=origin, u=u, v=v, cps=cps))

    # match consecutive frames with partial OT
    all_matches = []
    for k in range(len(frames) - 1):
        m = match_partial_ot(frames[k]["cps"], frames[k + 1]["cps"])
        print(f"matched {len(m)} / (t{frames[k]['ts']} -> t{frames[k+1]['ts']}) "
              f"[{len(frames[k]['cps'])} -> {len(frames[k+1]['cps'])} points]")
        all_matches.append(m)

    # -----------------------------------------------------------------
    # Build tracks: chains across as many consecutive frames as matched
    # -----------------------------------------------------------------
    # forward pointer: frame k index i -> frame k+1 index j (or None)
    fwd = []
    for k in range(len(frames) - 1):
        d = {i: j for (i, j) in all_matches[k]}
        fwd.append(d)

    tracks = []  # each: list of (frame_idx, point_idx)
    used_start = [set() for _ in frames]
    for k in range(len(frames)):
        for i in range(len(frames[k]["cps"])):
            if k > 0 and i in fwd[k - 1].values():
                continue  # already continues a track that started earlier
            chain = [(k, i)]
            cur_k, cur_i = k, i
            while cur_k < len(frames) - 1 and cur_i in fwd[cur_k]:
                nxt = fwd[cur_k][cur_i]
                chain.append((cur_k + 1, nxt))
                cur_k, cur_i = cur_k + 1, nxt
            tracks.append(chain)

    print(f"built {len(tracks)} track segments (including births/deaths)")

    # -----------------------------------------------------------------
    # VTK visualization
    # -----------------------------------------------------------------
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)

    kind_colors = {
        "source": (0.85, 0.1, 0.1),
        "sink": (0.1, 0.1, 0.85),
        "saddle": (0.1, 0.7, 0.1),
        "repelling focus": (0.9, 0.5, 0.0),
        "attracting focus": (0.0, 0.6, 0.6),
    }

    # (a) vector field glyphs per time layer, offset in z by t_idx*Z_STEP
    for f in frames:
        dims, spacing, origin, u, v = f["dims"], f["spacing"], f["origin"], f["u"], f["v"]
        nx, ny = dims[0], dims[1]
        z = f["t_idx"] * Z_STEP

        pts = vtk.vtkPoints()
        vecs = vtk.vtkFloatArray()
        vecs.SetNumberOfComponents(3)
        vecs.SetName("vec")
        mags = vtk.vtkFloatArray()
        mags.SetName("mag")

        for j in range(0, ny, GLYPH_STRIDE):
            for i in range(0, nx, GLYPH_STRIDE):
                x = origin[0] + i * spacing[0]
                y = origin[1] + j * spacing[1]
                uu, vv = float(u[j, i]), float(v[j, i])
                pts.InsertNextPoint(x, y, z)
                vecs.InsertNextTuple3(uu, vv, 0.0)
                mags.InsertNextValue((uu * uu + vv * vv) ** 0.5)

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
        glyph.SetScaleFactor(40.0)
        glyph.OrientOn()
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOn()
        mapper.SetScalarRange(0, 1)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.55)
        renderer.AddActor(actor)

        # faint plane / outline per layer for context
        outline = vtk.vtkOutlineSource()
        bounds = [origin[0], origin[0] + (nx - 1) * spacing[0],
                  origin[1], origin[1] + (ny - 1) * spacing[1], z, z]
        outline.SetBounds(bounds)
        om = vtk.vtkPolyDataMapper()
        om.SetInputConnection(outline.GetOutputPort())
        oa = vtk.vtkActor()
        oa.SetMapper(om)
        oa.GetProperty().SetColor(0.6, 0.6, 0.6)
        renderer.AddActor(oa)

    # (b) critical points as colored spheres
    for f in frames:
        z = f["t_idx"] * Z_STEP
        for (x, y, idx, kind) in f["cps"]:
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(x, y, z)
            sphere.SetRadius(2.2)
            sphere.SetPhiResolution(12)
            sphere.SetThetaResolution(12)
            sm = vtk.vtkPolyDataMapper()
            sm.SetInputConnection(sphere.GetOutputPort())
            sa = vtk.vtkActor()
            sa.SetMapper(sm)
            sa.GetProperty().SetColor(*kind_colors.get(kind, (0, 0, 0)))
            renderer.AddActor(sa)

    # (c) track lines connecting matched critical points across time
    track_points = vtk.vtkPoints()
    track_lines = vtk.vtkCellArray()
    track_colors = vtk.vtkUnsignedCharArray()
    track_colors.SetNumberOfComponents(3)
    track_colors.SetName("colors")

    rng = np.random.default_rng(0)
    for chain in tracks:
        if len(chain) < 2:
            continue
        color = (rng.integers(30, 220), rng.integers(30, 220), rng.integers(30, 220))
        ids = []
        for (k, i) in chain:
            x, y, idx, kind = frames[k]["cps"][i]
            z = frames[k]["t_idx"] * Z_STEP
            pid = track_points.InsertNextPoint(x, y, z)
            ids.append(pid)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(ids))
        for n, pid in enumerate(ids):
            line.GetPointIds().SetId(n, pid)
        track_lines.InsertNextCell(line)
        for _ in ids:
            track_colors.InsertNextTuple3(*color)

    track_pd = vtk.vtkPolyData()
    track_pd.SetPoints(track_points)
    track_pd.SetLines(track_lines)
    track_pd.GetPointData().SetScalars(track_colors)

    tube = vtk.vtkTubeFilter()
    tube.SetInputData(track_pd)
    tube.SetRadius(0.9)
    tube.SetNumberOfSides(10)
    tube.Update()

    tm = vtk.vtkPolyDataMapper()
    tm.SetInputConnection(tube.GetOutputPort())
    ta = vtk.vtkActor()
    ta.SetMapper(tm)
    renderer.AddActor(ta)

    # save track + critical point data to disk as well
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName("/workspace/cylinder_cp_tracks.vtp")
    writer.SetInputData(track_pd)
    writer.Write()

    cp_points = vtk.vtkPoints()
    cp_kind = vtk.vtkIntArray()
    cp_kind.SetName("kind_id")
    cp_time = vtk.vtkIntArray()
    cp_time.SetName("time_step")
    kind_ids = {"source": 0, "sink": 1, "saddle": 2, "repelling focus": 3, "attracting focus": 4}
    for f in frames:
        for (x, y, idx, kind) in f["cps"]:
            cp_points.InsertNextPoint(x, y, f["t_idx"] * Z_STEP)
            cp_kind.InsertNextValue(kind_ids.get(kind, -1))
            cp_time.InsertNextValue(f["ts"])
    cp_pd = vtk.vtkPolyData()
    cp_pd.SetPoints(cp_points)
    cp_pd.GetPointData().AddArray(cp_kind)
    cp_pd.GetPointData().AddArray(cp_time)
    cp_writer = vtk.vtkXMLPolyDataWriter()
    cp_writer.SetFileName("/workspace/cylinder_critical_points.vtp")
    cp_writer.SetInputData(cp_pd)
    cp_writer.Write()

    # camera + render offscreen -> PNG
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Azimuth(25)
    cam.Elevation(20)
    renderer.ResetCameraClippingRange()

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(1400, 1000)
    render_window.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()
    png = vtk.vtkPNGWriter()
    png.SetFileName("/workspace/cylinder_cp_tracking.png")
    png.SetInputConnection(w2i.GetOutputPort())
    png.Write()
    print("wrote /workspace/cylinder_cp_tracking.png")
    print("wrote /workspace/cylinder_cp_tracks.vtp")
    print("wrote /workspace/cylinder_critical_points.vtp")


if __name__ == "__main__":
    main()
