from paraview.simple import *
from paraview import servermanager
import os, re

OUT = '/workspace/ionization_renderings'
os.makedirs(OUT, exist_ok=True)

# Make TTK filters available to ParaView's Python interface.
LoadPlugin('/opt/conda/bin/plugins/TopologyToolKit/TopologyToolKit.so', remote=False, ns=globals())

files = sorted(
    [os.path.join('/workspace/Ionization', f) for f in os.listdir('/workspace/Ionization') if f.endswith('.vti')],
    key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))
reader = XMLImageDataReader(FileName=files)
reader.PointArrayStatus = ['Scalars_']
reader.UpdatePipeline()

# Original scalar field: use the middle Z slice so its scalar structure is visible.
view = CreateView('RenderView')
view.ViewSize = [1400, 900]
view.Background = [0.08, 0.08, 0.10]
view.OrientationAxesVisibility = 0
sl = Slice(Input=reader)
sl.SliceType = 'Plane'
sl.SliceType.Origin = [149.5, 61.5, 61.5]
sl.SliceType.Normal = [0.0, 0.0, 1.0]
srep = Show(sl, view)
ColorBy(srep, ('POINTS', 'Scalars_'))
lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Viridis (matplotlib)', True)
# Let ParaView use the scalar range provided by the current dataset.
srep.RescaleTransferFunctionToDataRange(True, False)
srep.SetScalarBarVisibility(view, True)
bar = GetScalarBar(lut, view)
bar.Title = 'Ionization scalar'
bar.ComponentTitle = ''
view.CameraPosition = [149.5, 61.5, 700]
view.CameraFocalPoint = [149.5, 61.5, 61.5]
view.CameraParallelScale = 150
Render()
SaveScreenshot(os.path.join(OUT, 'ionization_original_viridis.png'), view)

# Persistence simplification (absolute threshold 0.1), followed by a contour tree.
simp = TTKTopologicalSimplificationByPersistence(Input=reader)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PersistenceThreshold = 0.1
simp.ThresholdIsAbsolute = 1
simp.PairType = 0
simp.ThreadNumber = 0
tree = TTKContourTree(Input=simp)
tree.ScalarField = ['POINTS', 'Scalars_']
tree.ArcSampling = 0
tree.ThreadNumber = 0
tree.UpdatePipeline()

# TTK output ports: 0 = nodes, 1 = arcs, 2 = segmentation.
nodes = OutputPort(tree, 0)
arcs = OutputPort(tree, 1)

# Render the piecewise-linear arcs with thickness 1.
arep = Show(arcs, view)
arep.Representation = 'Surface'
ColorBy(arep, None)
arep.DiffuseColor = [0.72, 0.72, 0.72]
arep.LineWidth = 1.0

# Render each critical-point type independently, using the TTK CriticalType codes:
# 0=minima, 1=1-saddles, 2=2-saddles, 3=maxima.
critical_colors = {0: (0.10, 0.32, 1.00), 1: (1.0, 1.0, 1.0), 2: (1.0, 0.45, 0.0), 3: (1.0, 0.0, 0.0)}
for ctype, color in critical_colors.items():
    th = Threshold(Input=nodes)
    th.Scalars = ['POINTS', 'CriticalType']
    th.ThresholdMethod = 'Between'
    th.LowerThreshold = ctype
    th.UpperThreshold = ctype
    rep = Show(th, view)
    rep.Representation = 'Points'
    rep.PointSize = 2.0
    rep.DiffuseColor = color
    rep.AmbientColor = color
    rep.Ambient = 1.0

# TTK also labels a small number of degenerate critical vertices (type 4).
# They are shown white, consistent with saddle styling.
th = Threshold(Input=nodes)
th.Scalars = ['POINTS', 'CriticalType']
th.ThresholdMethod = 'Between'
th.LowerThreshold = 4
th.UpperThreshold = 4
rep = Show(th, view)
rep.Representation = 'Points'
rep.PointSize = 2.0
rep.DiffuseColor = [1.0, 1.0, 1.0]
rep.AmbientColor = [1.0, 1.0, 1.0]
rep.Ambient = 1.0

Hide(sl, view)
view.CameraPosition = [149.5, -520, 420]
view.CameraFocalPoint = [149.5, 61.5, 61.5]
view.CameraViewUp = [0, 0, 1]
view.CameraParallelScale = 420
Render()
SaveScreenshot(os.path.join(OUT, 'ionization_simplified_contour_tree.png'), view)

# Save the fully reproducible ParaView state and the derived contour-tree geometry.
SaveState(os.path.join(OUT, 'ionization_contour_tree.pvsm'))
SaveData(os.path.join(OUT, 'ionization_simplified_contour_tree_arcs.vtu'), proxy=arcs)
SaveData(os.path.join(OUT, 'ionization_simplified_contour_tree_vertices.vtu'), proxy=nodes)
print('Saved outputs to', OUT)
