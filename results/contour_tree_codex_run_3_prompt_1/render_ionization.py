from paraview.simple import *

# Ionization time series: ParaView recognizes the numbered VTI files as a file
# series when all are provided to the reader.
files = [f'/workspace/Ionization/Ionization{i}.vti' for i in range(1, 22)]
data = XMLImageDataReader(FileName=files)
data.PointArrayStatus = ['Scalars_']
data.UpdatePipeline()

# Render the first time step as a central scalar slice using viridis.
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
scene.AnimationTime = 0
view = CreateView('RenderView')
view.ViewSize = [1400, 900]
view.Background = [1, 1, 1]
sl = Slice(Input=data)
sl.SliceType = 'Plane'
sl.SliceType.Origin = [149.5, 61.5, 61.5]
sl.SliceType.Normal = [0, 0, 1]
sl.UpdatePipeline()
sv = Show(sl, view)
ColorBy(sv, ('POINTS', 'Scalars_'))
lut = GetColorTransferFunction('Scalars_')
lut.ApplyPreset('Viridis (matplotlib)', True)
sv.SetScalarBarVisibility(view, True)
sb = GetScalarBar(lut, view)
sb.Title = 'Scalars_'
sb.ComponentTitle = ''
view.CameraPosition = [149.5, 61.5, 650]
view.CameraFocalPoint = [149.5, 61.5, 61.5]
view.CameraViewUp = [0, 1, 0]
Render()
SaveScreenshot('/workspace/ionization_original_viridis.png', view)

# Persistence simplification, then construct the corresponding PL contour tree.
Hide(sl, view)
simp = TTKTopologicalSimplificationByPersistence(Input=data)
simp.InputArray = ['POINTS', 'Scalars_']
simp.PersistenceThreshold = 0.1
simp.ThresholdIsAbsolute = 1
simp.PairType = 0
simp.UpdatePipeline()

tree = TTKContourTree(Input=simp)
tree.ScalarField = ['POINTS', 'Scalars_']
tree.ArcSampling = 0
tree.UpdatePipeline()

# Port 1 is the contour-tree arc geometry; give each arc thickness 1.
arcs = OutputPort(tree, 1)
ar = Show(arcs, view)
ar.Representation = 'Surface'
ar.LineWidth = 1.0
ar.DiffuseColor = [0.15, 0.15, 0.15]

# Port 0 contains vertices.  Threshold the critical-type codes so each
# requested critical point class has its own explicit render color.
nodes = OutputPort(tree, 0)
colors = {
    0: [0.0, 0.25, 1.0],   # minima: blue
    1: [1.0, 1.0, 1.0],    # 1-saddles: white
    2: [1.0, 0.5, 0.0],    # 2-saddles: orange
    3: [1.0, 0.0, 0.0],    # maxima: red
}
for code, color in colors.items():
    crit = Threshold(Input=nodes)
    crit.Scalars = ['POINTS', 'CriticalType']
    crit.LowerThreshold = code
    crit.UpperThreshold = code
    glyph = Glyph(Input=crit, GlyphType='Sphere')
    glyph.GlyphType.Radius = 2.0
    glyph.GlyphType.ThetaResolution = 16
    glyph.GlyphType.PhiResolution = 16
    disp = Show(glyph, view)
    disp.DiffuseColor = color
    disp.AmbientColor = color
    disp.Ambient = 0.35

# Keep the tree in its native piecewise-linear embedding, looking down Z.
view.CameraPosition = [149.5, 61.5, 650]
view.CameraFocalPoint = [149.5, 61.5, 61.5]
view.CameraViewUp = [0, 1, 0]
view.OrientationAxesVisibility = 0
Render()
SaveScreenshot('/workspace/ionization_simplified_contour_tree.png', view)

# Persist all sources, properties, colors, and time-series input in a reusable
# ParaView state file.
SaveState('/workspace/ionization_contour_tree.pvsm')
