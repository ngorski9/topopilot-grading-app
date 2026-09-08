from paraview.simple import *
from paraview import servermanager
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

OUT = '/workspace/topology_output'
PLUGIN = '/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so'
LoadPlugin(PLUGIN, remote=False, ns=globals())

# Read and persistence-simplify the scalar field.  A relative threshold of 0.04
# is used (4% of the data range), matching TTK's normalized threshold setting.
source = XMLImageDataReader(registrationName='QMCPACK', FileName=['/workspace/QMCPACK.vti'])
simplified = TTKTopologicalSimplificationByPersistence(registrationName='Persistence simplification (0.04)', Input=source)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.04
simplified.ThresholdIsAbsolute = 0
simplified.UpdatePipeline()

# The four contours are scalar quantiles: each consecutive pair bounds 20% of
# the sampled volume (ties are resolved by NumPy's standard linear quantile).
sdata = servermanager.Fetch(simplified)
values = vtk_to_numpy(sdata.GetPointData().GetArray('Scalars_'))
isovalues = [float(np.quantile(values, q)) for q in (0.2, 0.4, 0.6, 0.8)]

contours = Contour(registrationName='Four equal-volume isocontours', Input=simplified)
contours.ContourBy = ['POINTS', 'Scalars_']
contours.Isosurfaces = isovalues
contours.PointMergeMethod = 'Uniform Binning'

# Extract all PL critical points after simplification. CriticalType values are
# TTK's Morse indices: 0=min, 1=1-saddle, 2=2-saddle, 3=max.
critical = TTKScalarFieldCriticalPoints(registrationName='PL critical points', Input=simplified)
critical.ScalarField = ['POINTS', 'Scalars_']
critical.Withvertexidentifiers = 1
critical.Withvertexscalars = 1
critical.UpdatePipeline()

def critical_subset(name, crit_type):
    t = Threshold(registrationName=name, Input=critical)
    t.Scalars = ['POINTS', 'CriticalType']
    t.LowerThreshold = crit_type
    t.UpperThreshold = crit_type
    t.ThresholdMethod = 'Between'
    return t

mins = critical_subset('Minima', 0)
s1 = critical_subset('1-saddles', 1)
s2 = critical_subset('2-saddles', 2)
maxs = critical_subset('Maxima', 3)
multi = critical_subset('Degenerate multi-saddles', 4)

view = CreateView('RenderView')
view.ViewSize = [1800, 1300]
view.Background = [0.055, 0.065, 0.085]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.CameraParallelProjection = 0

contour_display = Show(contours, view)
contour_display.Representation = 'Surface'
contour_display.DiffuseColor = [0.52, 0.78, 0.86]
contour_display.Opacity = 0.35
contour_display.Specular = 0.35
contour_display.SpecularPower = 30

def show_points(proxy, color):
    disp = Show(proxy, view)
    disp.Representation = 'Points'
    disp.DiffuseColor = color
    disp.PointSize = 13
    disp.RenderPointsAsSpheres = 1
    disp.AmbientColor = color
    disp.Ambient = 0.25
    disp.Specular = 0.5
    return disp

show_points(maxs, [0.94, 0.10, 0.10])
show_points(s2, [1.00, 0.48, 0.05])
show_points(s1, [1.00, 1.00, 1.00])
show_points(multi, [1.00, 1.00, 1.00])
show_points(mins, [0.12, 0.36, 1.00])

title = Text(registrationName='Title')
title.Text = 'QMCPACK — persistence-simplified topology (threshold = 0.04)'
td = Show(title, view)
td.WindowLocation = 'Upper Center'
td.Color = [0.94, 0.96, 1.0]
td.FontSize = 20
td.Bold = 1

legend = Text(registrationName='Critical point legend')
legend.Text = 'Critical points:  red = maxima   |   orange = 2-saddles   |   white = 1-saddles / degenerate multi-saddles   |   blue = minima\n4 translucent isocontours at 20%, 40%, 60%, and 80% volume quantiles'
ld = Show(legend, view)
ld.WindowLocation = 'Lower Center'
ld.Color = [0.88, 0.90, 0.94]
ld.FontSize = 15

ResetCamera(view)
view.CameraPosition = [205, -240, 200]
view.CameraFocalPoint = [34, 34, 57]
view.CameraViewUp = [0, 0, 1]
view.CameraViewAngle = 28
Render(view)
SaveScreenshot(OUT + '/QMCPACK_persistence_0.04.png', view, ImageResolution=[1800, 1300])
SaveState(OUT + '/QMCPACK_persistence_0.04.pvsm')

# Persist numerical details and the actual filtered critical-point geometry.
np.savetxt(OUT + '/isovalues.txt', np.asarray(isovalues), header='20%, 40%, 60%, 80% quantile isovalues')
for name, proxy in [('minima', mins), ('1_saddles', s1), ('2_saddles', s2), ('maxima', maxs), ('multi_saddles', multi)]:
    SaveData(OUT + '/critical_points_' + name + '.vtu', proxy=proxy)
print('isovalues:', isovalues)
print('saved:', OUT + '/QMCPACK_persistence_0.04.png')
