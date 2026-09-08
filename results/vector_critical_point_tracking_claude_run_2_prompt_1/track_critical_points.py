"""
Track critical points of the time-dependent 'cylinder' vector field
across 3 timesteps using Partial Optimal Transport (POT).

Pipeline:
  1. Extract critical points (source/sink/saddle) of the piecewise-linear
     vector field (u, v) for timesteps 1, 2, 3 (extract_critical_points.py).
  2. Match critical points between consecutive timesteps with
     ot.partial.partial_wasserstein -- "partial" because the number of
     critical points can change between timesteps (birth/death), so we
     only transport as much mass as the smaller point set can absorb and
     leave the rest unmatched.
  3. Chain the pairwise matches into trajectories and write:
       - critical_points.vtp   (all critical points, all timesteps)
       - trajectories.vtp      (polylines through matched points)
"""
import numpy as np
import vtk
from vtk.util import numpy_support as vns
import ot

from extract_critical_points import extract_critical_points

FILES = ['cylinder/cylinder1.vti', 'cylinder/cylinder2.vti', 'cylinder/cylinder3.vti']
TYPE_MISMATCH_PENALTY = 400.0   # added to squared-distance cost when types differ
MATCH_MASS_FRACTION = 0.92      # fraction of min(n_t, n_t+1) mass to transport
GAMMA_THRESHOLD = 0.3           # minimal transported mass to call it a match


def match_partial_ot(pts_a, types_a, pts_b, types_b):
    na, nb = len(pts_a), len(pts_b)
    d2 = ((pts_a[:, None, :2] - pts_b[None, :, :2]) ** 2).sum(-1)
    mismatch = (types_a[:, None] != types_b[None, :]).astype(float)
    M = d2 + TYPE_MISMATCH_PENALTY * mismatch

    a = np.ones(na)
    b = np.ones(nb)
    m = MATCH_MASS_FRACTION * min(na, nb)

    gamma = ot.partial.partial_wasserstein(a, b, M, m=m)

    matches = []  # (i, j)
    for i in range(na):
        j = np.argmax(gamma[i])
        if gamma[i, j] > GAMMA_THRESHOLD:
            matches.append((i, j))
    return matches, gamma


def main():
    all_pts, all_types = [], []
    for f in FILES:
        pts, types = extract_critical_points(f)
        all_pts.append(pts)
        all_types.append(types)
        print(f'{f}: {len(pts)} critical points '
              f'(saddle={np.sum(types==0)}, source={np.sum(types==1)}, sink={np.sum(types==2)})')

    pairwise_matches = []
    for t in range(len(FILES) - 1):
        matches, gamma = match_partial_ot(all_pts[t], all_types[t], all_pts[t + 1], all_types[t + 1])
        pairwise_matches.append(matches)
        print(f'timestep {t+1} -> {t+2}: {len(matches)} matched critical points '
              f'out of {len(all_pts[t])} / {len(all_pts[t+1])}')

    # ---- write all critical points (per timestep) ----
    pts_poly = vtk.vtkAppendPolyData()
    for t, (pts, types) in enumerate(zip(all_pts, all_types)):
        pd = vtk.vtkPolyData()
        vpts = vtk.vtkPoints()
        for p in pts:
            vpts.InsertNextPoint(p[0], p[1], p[2])
        pd.SetPoints(vpts)
        verts = vtk.vtkCellArray()
        for i in range(len(pts)):
            verts.InsertNextCell(1, [i])
        pd.SetVerts(verts)

        arr_type = vns.numpy_to_vtk(types.astype(np.int32))
        arr_type.SetName('CriticalType')
        pd.GetPointData().AddArray(arr_type)

        arr_ts = vns.numpy_to_vtk(np.full(len(pts), t, dtype=np.int32))
        arr_ts.SetName('TimeStep')
        pd.GetPointData().AddArray(arr_ts)

        pts_poly.AddInputData(pd)
    pts_poly.Update()

    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName('critical_points.vtp')
    w.SetInputData(pts_poly.GetOutput())
    w.Write()
    print('wrote critical_points.vtp')

    # ---- chain matches t1->t2->t3 into trajectories ----
    map01 = dict(pairwise_matches[0])
    map12 = dict(pairwise_matches[1])

    chains = []  # list of (idx_t0_or_None, idx_t1_or_None, idx_t2_or_None)
    used0, used1 = set(), set()
    for i0, i1 in map01.items():
        used0.add(i0)
        used1.add(i1)
        i2 = map12.get(i1, None)
        chains.append((i0, i1, i2))
    # t1 points not matched forward (deaths after t1, or newly matched only from t1->t2 miss)
    for i1 in range(len(all_pts[1])):
        if i1 not in used1:
            i2 = map12.get(i1, None)
            chains.append((None, i1, i2))
    # births at t2 (unused) already covered above when i1 not in used1 and i2 present
    # pure singleton unmatched points at t0 (deaths at t0->t1)
    for i0 in range(len(all_pts[0])):
        if i0 not in used0:
            chains.append((i0, None, None))
    # pure singleton unmatched points at t2 (births at t1->t2, already listed) -- also unmatched t2 with no t1 link
    used2_via_chain = set(i2 for (_, _, i2) in chains if i2 is not None)
    for i2 in range(len(all_pts[2])):
        if i2 not in used2_via_chain:
            chains.append((None, None, i2))

    lines_poly = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    traj_len_arr = []
    traj_type_arr = []
    point_type_arr = []
    point_id = 0
    for chain in chains:
        ids = [i for i in chain if i is not None]
        n_present = len(ids)
        if n_present < 1:
            continue
        seq_ids = []
        for t, idx in enumerate(chain):
            if idx is None:
                continue
            p = all_pts[t][idx]
            vpts.InsertNextPoint(p[0], p[1], p[2])
            point_type_arr.append(all_types[t][idx])
            seq_ids.append(point_id)
            point_id += 1
        if len(seq_ids) >= 2:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(seq_ids))
            for k, pid in enumerate(seq_ids):
                line.GetPointIds().SetId(k, pid)
            lines.InsertNextCell(line)
            traj_len_arr.append(len(seq_ids))
        else:
            v = vtk.vtkVertex()
            v.GetPointIds().SetId(0, seq_ids[0])
            lines.InsertNextCell(v)
            traj_len_arr.append(1)

    lines_poly.SetPoints(vpts)
    lines_poly.SetLines(lines)

    arr = vns.numpy_to_vtk(np.array(point_type_arr, dtype=np.int32))
    arr.SetName('CriticalType')
    lines_poly.GetPointData().AddArray(arr)

    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName('trajectories.vtp')
    w.SetInputData(lines_poly)
    w.Write()
    print('wrote trajectories.vtp')

    n_full = sum(1 for c in chains if c[0] is not None and c[1] is not None and c[2] is not None)
    n_partial = sum(1 for c in chains if sum(x is not None for x in c) == 2)
    n_singleton = sum(1 for c in chains if sum(x is not None for x in c) == 1)
    print(f'trajectories: {n_full} tracked across all 3 timesteps, '
          f'{n_partial} partial (birth/death), {n_singleton} singleton (unmatched)')


if __name__ == '__main__':
    main()
