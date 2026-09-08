from paraview.simple import *
from paraview import servermanager
import glob

# A numerically sorted multi-file VTI reader is interpreted by ParaView as a
# time series.  The initial displayed time is the first Ionization step.
files = sorted(glob.glob('/workspace/Ionization/Ionization*.vti'),
               key=lambda p: int(p.rsplit('Ionization', 1)[1].split('.vti')[0]))
ionization = XMLImageDataReader(registrationName='Ionization (time series)', FileName=files)
ionization.PointArrayStatus = ['Scalars_']

# Persistence simplification is performed before constructing the tree.
simplified = TTKTopologicalSimplificationByPersistence(
    registrationName='Persistence simplification (0.1)', Input=ionization)
simplified.InputArray = ['POINTS', 'Scalars_']
simplified.PersistenceThreshold = 0.1
simplified.ThresholdIsAbsolute = 1

contourTree = TTKContourTree(registrationName='Simplified piecewise-linear contour tree', Input=simplified)
contourTree.ScalarField = ['POINTS', 'Scalars_']
contourTree.ArcSampling = 0

view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1600, 900]
view.Background = [0.08, 0.08, 0.10]

# Raw scalar field, viridis.  A semi-transparent volume keeps the tree legible.
rawDisplay = Show(ionization, view, 'UniformGridRepresentation')
ColorBy(rawDisplay, ('POINTS', 'Scalars_'))
rawLUT = GetColorTransferFunction('Scalars_')
rawLUT.ApplyPreset('Viridis (matplotlib)', True)
rawDisplay.SetRepresentationType('Volume')
rawDisplay.Opacity = 0.20
rawDisplay.RescaleTransferFunctionToDataRange(True, False)
rawDisplay.Visibility = 1

# TTK ports: 0 = critical points/nodes, 1 = contour-tree arcs.
treeNodes = OutputPort(contourTree, 0)
treeArcs = OutputPort(contourTree, 1)

arcGeometry = ExtractSurface(registrationName='Contour-tree arc geometry', Input=treeArcs)
edgeTubes = Tube(registrationName='Contour-tree edges (radius 1)', Input=arcGeometry)
edgeTubes.Radius = 1.0
edgeTubes.NumberofSides = 12
arcDisplay = Show(edgeTubes, view, 'GeometryRepresentation')
arcDisplay.DiffuseColor = [0.82, 0.82, 0.82]
arcDisplay.LineWidth = 2.0

# Split node types using the TTK critical-type convention:
# 0=min, 1=1-saddle, 2=2-saddle, 3=max.
def critical_points(name, value, color):
    f = Threshold(registrationName=name, Input=treeNodes)
    f.Scalars = ['POINTS', 'CriticalType']
    f.LowerThreshold = value
    f.UpperThreshold = value
    g = Glyph(registrationName=name + ' vertices (radius 2)', Input=f,
              GlyphType='Sphere')
    g.GlyphType.Radius = 2.0
    g.ScaleArray = ['POINTS', 'No scale array']
    g.ScaleFactor = 1.0
    d = Show(g, view, 'GeometryRepresentation')
    d.DiffuseColor = color
    return g

minima = critical_points('Minima (blue)', 0, [0.0, 0.25, 1.0])
saddle1 = critical_points('1-saddles (white)', 1, [1.0, 1.0, 1.0])
saddle2 = critical_points('2-saddles (orange)', 2, [1.0, 0.5, 0.0])
maxima = critical_points('Maxima (red)', 3, [1.0, 0.0, 0.0])

view.Update()
Render()
view.ResetCamera()
Render()
SaveState('/workspace/Ionization_contour_tree.pvsm')
SaveScreenshot('/workspace/Ionization_contour_tree.png', view, ImageResolution=[1600,900])
