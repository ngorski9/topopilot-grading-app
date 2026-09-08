import numpy as np
import vtk
from vtk.util import numpy_support as ns
from paraview.simple import *

# ---------------------------------------------------------------------------
# 1. Load Ocean.vti
# ---------------------------------------------------------------------------
reader = XMLImageDataReader(FileName=['/workspace/Ocean.vti'])
reader.PointArrayStatus = ['A', 'B', 'C', 'D']

vtkReader = vtk.vtkXMLImageDataReader()
vtkReader.SetFileName('/workspace/Ocean.vti')
vtkReader.Update()
img = vtkReader.GetOutput()
dims = img.GetDimensions()
nx, ny = dims[0], dims[1]
spacing = img.GetSpacing()
origin = img.GetOrigin()

pd = img.GetPointData()
A = ns.vtk_to_numpy(pd.GetArray('A')).reshape(ny, nx)
B = ns.vtk_to_numpy(pd.GetArray('B')).reshape(ny, nx)
C = ns.vtk_to_numpy(pd.GetArray('C')).reshape(ny, nx)
D = ns.vtk_to_numpy(pd.GetArray('D')).reshape(ny, nx)

# deviator components of the 2x2 (possibly asymmetric) tensor field
E = (A - D) / 2.0
F = (B + C) / 2.0
G = (B - C) / 2.0
Delta = E ** 2 + F ** 2 - G ** 2  # >0: real distinct eigenvectors, <0: complex eigenvalues

# ---------------------------------------------------------------------------
# 2. Add derived arrays (E, F, G, Delta) to the pipeline via Calculator chain
# ---------------------------------------------------------------------------
calcE = Calculator(Input=reader)
calcE.ResultArrayName = 'E'
calcE.Function = '(A-D)/2'

calcF = Calculator(Input=calcE)
calcF.ResultArrayName = 'F'
calcF.Function = '(B+C)/2'

calcG = Calculator(Input=calcF)
calcG.ResultArrayName = 'G'
calcG.Function = '(B-C)/2'

calcDelta = Calculator(Input=calcG)
calcDelta.ResultArrayName = 'Delta'
calcDelta.Function = 'E^2+F^2-G^2'

calcPartition = Calculator(Input=calcDelta)
calcPartition.ResultArrayName = 'Partition'
calcPartition.Function = 'Delta/abs(Delta)'  # +1 real distinct eig, -1 complex eig

# ---------------------------------------------------------------------------
# 3. Eigenvector partition, resampled to a coarse grid: 10 pixels per square
# ---------------------------------------------------------------------------
res_x = max(2, int(round((nx - 1) / 10.0)) + 1)
res_y = max(2, int(round((ny - 1) / 10.0)) + 1)
partitionResample = ResampleToImage(Input=calcPartition)
partitionResample.SamplingDimensions = [res_x, res_y, 1]
partitionResample.SamplingBounds = [origin[0], origin[0] + (nx - 1) * spacing[0],
                                     origin[1], origin[1] + (ny - 1) * spacing[1],
                                     0.0, 0.0]

partitionSurface = partitionResample

# ---------------------------------------------------------------------------
# 4. Degenerate point extraction (E=0 and F=0), classified as wedge/trisector
#    delta = dE/dx*dF/dy - dE/dy*dF/dx  -> >0 wedge, <0 trisector
# ---------------------------------------------------------------------------
dEdx, dEdy = np.gradient(E, spacing[0], spacing[1], axis=(1, 0))
dFdx, dFdy = np.gradient(F, spacing[0], spacing[1], axis=(1, 0))

points = vtk.vtkPoints()
ptype = vtk.vtkIntArray()
ptype.SetName('DegenerateType')  # 0 = wedge, 1 = trisector

for j in range(ny - 1):
    for i in range(nx - 1):
        e00, e10, e01, e11 = E[j, i], E[j, i + 1], E[j + 1, i], E[j + 1, i + 1]
        f00, f10, f01, f11 = F[j, i], F[j, i + 1], F[j + 1, i], F[j + 1, i + 1]
        if not (min(e00, e10, e01, e11) < 0 < max(e00, e10, e01, e11)):
            continue
        if not (min(f00, f10, f01, f11) < 0 < max(f00, f10, f01, f11)):
            continue

        # bilinear solve for E(s,t)=0, F(s,t)=0 on unit square via sampling + Newton
        def bilinear(v00, v10, v01, v11, s, t):
            return (v00 * (1 - s) * (1 - t) + v10 * s * (1 - t) +
                    v01 * (1 - s) * t + v11 * s * t)

        s, t = 0.5, 0.5
        converged = False
        for _ in range(30):
            e_val = bilinear(e00, e10, e01, e11, s, t)
            f_val = bilinear(f00, f10, f01, f11, s, t)
            de_ds = (e10 - e00) * (1 - t) + (e11 - e01) * t
            de_dt = (e01 - e00) * (1 - s) + (e11 - e10) * s
            df_ds = (f10 - f00) * (1 - t) + (f11 - f01) * t
            df_dt = (f01 - f00) * (1 - s) + (f11 - f10) * s
            det = de_ds * df_dt - de_dt * df_ds
            if abs(det) < 1e-12:
                break
            ds = (-e_val * df_dt + f_val * de_dt) / det
            dt = (-f_val * de_ds + e_val * df_ds) / det
            s += ds
            t += dt
            if abs(ds) < 1e-8 and abs(dt) < 1e-8:
                converged = True
                break
        if not converged or not (0.0 <= s <= 1.0 and 0.0 <= t <= 1.0):
            continue

        x = origin[0] + (i + s) * spacing[0]
        y = origin[1] + (j + t) * spacing[1]

        dEdx_p = dEdx[j, i] * (1 - s) * (1 - t) + dEdx[j, i + 1] * s * (1 - t) + \
            dEdx[j + 1, i] * (1 - s) * t + dEdx[j + 1, i + 1] * s * t
        dEdy_p = dEdy[j, i] * (1 - s) * (1 - t) + dEdy[j, i + 1] * s * (1 - t) + \
            dEdy[j + 1, i] * (1 - s) * t + dEdy[j + 1, i + 1] * s * t
        dFdx_p = dFdx[j, i] * (1 - s) * (1 - t) + dFdx[j, i + 1] * s * (1 - t) + \
            dFdx[j + 1, i] * (1 - s) * t + dFdx[j + 1, i + 1] * s * t
        dFdy_p = dFdy[j, i] * (1 - s) * (1 - t) + dFdy[j, i + 1] * s * (1 - t) + \
            dFdy[j + 1, i] * (1 - s) * t + dFdy[j + 1, i + 1] * s * t

        delta = dEdx_p * dFdy_p - dEdy_p * dFdx_p
        points.InsertNextPoint(x, y, 0.0)
        ptype.InsertNextValue(0 if delta > 0 else 1)

degPoly = vtk.vtkPolyData()
degPoly.SetPoints(points)
degPoly.GetPointData().AddArray(ptype)

nWedge = sum(1 for k in range(ptype.GetNumberOfTuples()) if ptype.GetValue(k) == 0)
nTrisector = ptype.GetNumberOfTuples() - nWedge
print('Degenerate points found: %d wedges, %d trisectors' % (nWedge, nTrisector))

def make_poly(indices):
    pts = vtk.vtkPoints()
    for k in indices:
        pts.InsertNextPoint(points.GetPoint(k))
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    return poly

wedge_idx = [k for k in range(ptype.GetNumberOfTuples()) if ptype.GetValue(k) == 0]
trisector_idx = [k for k in range(ptype.GetNumberOfTuples()) if ptype.GetValue(k) == 1]

wedgePoly = make_poly(wedge_idx)
trisectorPoly = make_poly(trisector_idx)

wedgeSource = TrivialProducer()
wedgeSource.GetClientSideObject().SetOutput(wedgePoly)
trisectorSource = TrivialProducer()
trisectorSource.GetClientSideObject().SetOutput(trisectorPoly)

wedgeGlyph = Glyph(Input=wedgeSource, GlyphType='Sphere')
wedgeGlyph.GlyphType.Radius = 1.0
wedgeGlyph.ScaleFactor = 1.0
wedgeGlyph.GlyphMode = 'All Points'

trisectorGlyph = Glyph(Input=trisectorSource, GlyphType='Sphere')
trisectorGlyph.GlyphType.Radius = 1.0
trisectorGlyph.ScaleFactor = 1.0
trisectorGlyph.GlyphMode = 'All Points'

# ---------------------------------------------------------------------------
# 5. Rendering
# ---------------------------------------------------------------------------
view = GetActiveViewOrCreate('RenderView')
view.ViewSize = [1000, 1000]

partDisplay = Show(partitionSurface, view)
partDisplay.Representation = 'Surface'
ColorBy(partDisplay, ('POINTS', 'Partition'))
partLUT = GetColorTransferFunction('Partition')
partLUT.RGBPoints = [-1.0, 0.85, 0.85, 0.85,
                      -0.0001, 0.85, 0.85, 0.85,
                      0.0001, 0.55, 0.75, 0.95,
                      1.0, 0.55, 0.75, 0.95]
partLUT.ColorSpace = 'RGB'
partDisplay.SetScalarBarVisibility(view, False)

wedgeDisplay = Show(wedgeGlyph, view)
wedgeDisplay.Representation = 'Surface'
wedgeDisplay.AmbientColor = [1.0, 1.0, 1.0]
wedgeDisplay.DiffuseColor = [1.0, 1.0, 1.0]
wedgeDisplay.EdgeColor = [0.0, 0.0, 0.0]
wedgeDisplay.LineWidth = 1.0

trisectorDisplay = Show(trisectorGlyph, view)
trisectorDisplay.Representation = 'Surface'
trisectorDisplay.AmbientColor = [1.0, 0.41, 0.71]
trisectorDisplay.DiffuseColor = [1.0, 0.41, 0.71]

view.ResetCamera()
view.CameraPosition = [50, 50, 250]
view.CameraFocalPoint = [50, 50, 0]
view.CameraViewUp = [0, 1, 0]
view.Background = [1, 1, 1]

Render(view)
SaveScreenshot('/workspace/Ocean_eigenvector_partition.png', view, ImageResolution=[1000, 1000])
SaveState('/workspace/Ocean_eigenvector_partition.pvsm')
print('Wrote /workspace/Ocean_eigenvector_partition.png and .pvsm')
