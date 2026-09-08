import glob, os, re
from paraview.simple import *
from paraview import servermanager as sm
from vtkmodules.util.numpy_support import vtk_to_numpy
import numpy as np
import ot  # Python Optimal Transport

LoadDistributedPlugin('TopologyToolKit', ns=globals())

base = '/workspace/cloud'
files = glob.glob(os.path.join(base, 'cloud*.vti'))
files.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group()))
N_STEPS = 3
files = files[:N_STEPS]
print('Using files:', files)

PERSISTENCE_THRESHOLD = 0.5

def extract_maxima(fname):
    r = XMLImageDataReader(FileName=[fname])
    r.UpdatePipeline()

    pd = TTKPersistenceDiagram(Input=r)
    pd.ScalarField = ['POINTS', 'Scalars_']
    pd.UpdatePipeline()

    data = sm.Fetch(pd)
    pdata = data.GetPointData()
    cdata = data.GetCellData()

    coords = vtk_to_numpy(pdata.GetArray('Coordinates'))
    crit = vtk_to_numpy(pdata.GetArray('CriticalType'))
    scalars_pt = None
    if pdata.GetArray('Scalars_') is not None:
        scalars_pt = vtk_to_numpy(pdata.GetArray('Scalars_'))

    persistence = vtk_to_numpy(cdata.GetArray('Persistence'))
    pair_type = vtk_to_numpy(cdata.GetArray('PairType'))

    ncells = data.GetNumberOfCells()
    max_positions = []
    max_persistence = []
    for c in range(ncells):
        if pair_type[c] != 1:            # keep saddle-maximum pairs only
            continue
        if persistence[c] <= PERSISTENCE_THRESHOLD:   # persistence simplification
            continue
        cell = data.GetCell(c)
        pids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        # pick the point that is the maximum (CriticalType == 3)
        max_pid = None
        for pid in pids:
            if crit[pid] == 3:
                max_pid = pid
                break
        if max_pid is None:
            continue
        max_positions.append(coords[max_pid])
        max_persistence.append(persistence[c])

    return r, np.array(max_positions), np.array(max_persistence)

readers = []
maxima_positions = []
maxima_persistence = []
for f in files:
    r, pos, pers = extract_maxima(f)
    readers.append(r)
    maxima_positions.append(pos)
    maxima_persistence.append(pers)
    print(f, '-> maxima after persistence simplification (>%.1f):' % PERSISTENCE_THRESHOLD, len(pos))

# ---- Track maxima across timesteps using Earth Mover's Distance (optimal transport) ----
tracks = []  # list of (t0_idx, t1_idx) matched index pairs per consecutive timestep pair
for t in range(len(files) - 1):
    A = maxima_positions[t]
    B = maxima_positions[t + 1]
    wA = maxima_persistence[t] / maxima_persistence[t].sum()
    wB = maxima_persistence[t + 1] / maxima_persistence[t + 1].sum()
    M = ot.dist(A, B, metric='euclidean')  # cost matrix
    M /= M.max()
    G = ot.emd(wA, wB, M)                  # earth mover's distance transport plan
    match = G.argmax(axis=1)               # for each point in A, most-matched point in B
    tracks.append(match)
    print('timestep', t, '->', t + 1, ': matched', len(match), 'maxima via EMD')

# build trajectories of length N_STEPS by chaining matches, starting from every maximum at t=0
trajectories = []
for i in range(len(maxima_positions[0])):
    traj = [maxima_positions[0][i]]
    idx = i
    ok = True
    for t in range(len(files) - 1):
        idx = tracks[t][idx]
        if idx >= len(maxima_positions[t + 1]):
            ok = False
            break
        traj.append(maxima_positions[t + 1][idx])
    if ok:
        trajectories.append(np.array(traj))

print('Total tracked maxima trajectories across %d timesteps:' % N_STEPS, len(trajectories))

np.save('/workspace/trajectories.npy', np.array(trajectories))
for i, f in enumerate(files):
    np.save(f'/workspace/maxima_t{i}.npy', maxima_positions[i])
np.save('/workspace/maxima_pers_t0.npy', maxima_persistence[0])

print('DONE_EXTRACT')
