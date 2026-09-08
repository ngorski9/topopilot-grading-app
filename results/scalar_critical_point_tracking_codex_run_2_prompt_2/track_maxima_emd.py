"""Extract and track local maxima with Wasserstein-1 / earth mover's distance."""
from pathlib import Path
import numpy as np
import ot
from scipy.ndimage import maximum_filter
from vtkmodules.vtkIOXML import vtkXMLImageDataReader, vtkXMLPolyDataWriter
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints, vtkIntArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray

ROOT = Path('/workspace'); OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

def load(i):
    r = vtkXMLImageDataReader(); r.SetFileName(str(ROOT/'cloud'/f'cloud{i}.vti')); r.Update()
    data = r.GetOutput(); dims = data.GetDimensions()
    values = vtk_to_numpy(data.GetPointData().GetArray('Scalars_')).reshape((dims[1], dims[0]))
    return data, values

def maxima(data, values, n=16):
    # Strict, separated 2-D local maxima; retain strongest 16 for balanced EMD.
    mask = (values == maximum_filter(values, size=9, mode='nearest')) & (values > np.percentile(values, 80))
    candidates = np.argwhere(mask)
    candidates = candidates[np.argsort(values[mask])[::-1]]
    selected = []
    for y, x in candidates:
        if all((x-px)**2 + (y-py)**2 >= 12**2 for py, px in selected):
            selected.append((y, x))
        if len(selected) == n: break
    origin, spacing = data.GetOrigin(), data.GetSpacing()
    pts = np.array([[origin[0]+x*spacing[0], origin[1]+y*spacing[1], 0.] for y,x in selected])
    vals = np.array([values[y,x] for y,x in selected])
    return pts, vals

datasets, images = zip(*(load(i) for i in (1,2,3)))
sets = [maxima(d, im) for d, im in zip(datasets, images)]

# Equal masses and p=1 transport: POT's emd solves the exact earth mover plan.
plans = []
costs = []
for (p0, v0), (p1, v1) in zip(sets[:-1], sets[1:]):
    f0 = np.c_[p0[:,:2]/255., v0[:,None]/50.]
    f1 = np.c_[p1[:,:2]/255., v1[:,None]/50.]
    cost = ot.dist(f0, f1, metric='euclidean')
    plans.append(ot.emd(np.ones(len(p0))/len(p0), np.ones(len(p1))/len(p1), cost))
    costs.append(cost)

# Make each EMD assignment a piecewise-linear trajectory t=0 -> 1 -> 2.
matches01 = plans[0].argmax(axis=1)
matches12 = plans[1].argmax(axis=1)
points = vtkPoints(); lines = vtkCellArray(); timestep = vtkIntArray(); timestep.SetName('TimeStep')
track_id = vtkIntArray(); track_id.SetName('TrackingId')
for i, j in enumerate(matches01):
    k = matches12[j]
    line = vtkCellArray()
    ids = []
    for t, p in enumerate((sets[0][0][i], sets[1][0][j], sets[2][0][k])):
        pid = points.InsertNextPoint(*p); ids.append(pid); timestep.InsertNextValue(t); track_id.InsertNextValue(i)
    lines.InsertNextCell(3, ids)
poly = vtkPolyData(); poly.SetPoints(points); poly.SetLines(lines)
poly.GetPointData().AddArray(timestep); poly.GetPointData().AddArray(track_id)
w = vtkXMLPolyDataWriter(); w.SetFileName(str(OUT/'cloud_maxima_tracking_emd_3steps.vtp')); w.SetInputData(poly); w.Write()
np.savetxt(OUT/'emd_transport_costs.txt', np.array([np.sum(p*c) for p, c in zip(plans, costs)]), header='Wasserstein-1 EMD costs (t0->t1, t1->t2)')
print(f'Created {len(matches01)} piecewise-linear maxima tracks using POT EMD/Wasserstein-1.')
