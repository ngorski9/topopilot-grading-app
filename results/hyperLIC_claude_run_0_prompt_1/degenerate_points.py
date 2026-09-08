"""
Compute degenerate points (trisectors and wedges) of the symmetric 2x2
piecewise-linear tensor field (components A, B, D forming [[A,B],[B,D]])
stored in brain.vti, then render the field together with the degenerate
points using ParaView. Trisectors are drawn in pink, wedges in white,
both as spheres of radius 1.
"""

import vtk
import numpy as np
from paraview.simple import *

INPUT_FILE = "./brain.vti"


def compute_degenerate_points(filename):
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()
    img = reader.GetOutput()

    dims = img.GetDimensions()
    nx, ny = dims[0], dims[1]
    origin = np.array(img.GetOrigin())
    spacing = np.array(img.GetSpacing())

    pd = img.GetPointData()
    A = np.array([pd.GetArray("A").GetValue(i) for i in range(pd.GetArray("A").GetNumberOfTuples())])
    B = np.array([pd.GetArray("B").GetValue(i) for i in range(pd.GetArray("B").GetNumberOfTuples())])
    D = np.array([pd.GetArray("D").GetValue(i) for i in range(pd.GetArray("D").GetNumberOfTuples())])

    # delta-vector field whose zeros are the tensor degenerate points:
    #   delta1 = A - D , delta2 = 2*B
    delta1 = A - D
    delta2 = 2.0 * B

    def pid(ix, iy):
        return iy * nx + ix

    def point_xy(ix, iy):
        return origin[:2] + spacing[:2] * np.array([ix, iy])

    trisectors = []
    wedges = []

    # Split each image-data quad cell into 2 triangles and look for a zero
    # of the linear (delta1, delta2) field within each triangle.
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            p00 = pid(ix, iy)
            p10 = pid(ix + 1, iy)
            p01 = pid(ix, iy + 1)
            p11 = pid(ix + 1, iy + 1)

            triangles = [
                (p00, p10, p11),
                (p00, p11, p01),
            ]

            for (i1, i2, i3) in triangles:
                d1 = np.array([delta1[i1], delta1[i2], delta1[i3]])
                d2 = np.array([delta2[i1], delta2[i2], delta2[i3]])

                # Solve for barycentric weights (w1, w2, w3=1-w1-w2) such
                # that the interpolated (delta1, delta2) is zero.
                m00 = d1[0] - d1[2]
                m01 = d1[1] - d1[2]
                m10 = d2[0] - d2[2]
                m11 = d2[1] - d2[2]
                det = m00 * m11 - m01 * m10

                if det == 0.0:
                    continue

                rhs0 = -d1[2]
                rhs1 = -d2[2]

                w1 = (m11 * rhs0 - m01 * rhs1) / det
                w2 = (-m10 * rhs0 + m00 * rhs1) / det
                w3 = 1.0 - w1 - w2

                eps = -1e-9
                if w1 < eps or w2 < eps or w3 < eps:
                    continue

                xy1 = point_xy(ix, iy) if i1 == p00 else (
                    point_xy(ix + 1, iy) if i1 == p10 else (
                        point_xy(ix, iy + 1) if i1 == p01 else point_xy(ix + 1, iy + 1)))
                xy2 = point_xy(ix, iy) if i2 == p00 else (
                    point_xy(ix + 1, iy) if i2 == p10 else (
                        point_xy(ix, iy + 1) if i2 == p01 else point_xy(ix + 1, iy + 1)))
                xy3 = point_xy(ix, iy) if i3 == p00 else (
                    point_xy(ix + 1, iy) if i3 == p10 else (
                        point_xy(ix, iy + 1) if i3 == p01 else point_xy(ix + 1, iy + 1)))

                pos = w1 * xy1 + w2 * xy2 + w3 * xy3

                # Classification via the sign of the determinant of the
                # (delta1, delta2) gradient over the triangle:
                #   det < 0 -> trisector (Poincare index -1/2)
                #   det > 0 -> wedge     (Poincare index +1/2)
                if det < 0:
                    trisectors.append(pos)
                else:
                    wedges.append(pos)

    return np.array(trisectors), np.array(wedges)


def make_polydata(points_xy):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    for (x, y) in points_xy:
        pts.InsertNextPoint(x, y, 0.0)
    poly.SetPoints(pts)

    verts = vtk.vtkCellArray()
    for i in range(pts.GetNumberOfPoints()):
        verts.InsertNextCell(1, [i])
    poly.SetVerts(verts)
    return poly


trisector_pts, wedge_pts = compute_degenerate_points(INPUT_FILE)
print("Trisectors found:", len(trisector_pts))
print("Wedges found:", len(wedge_pts))

trisector_poly = make_polydata(trisector_pts) if len(trisector_pts) else vtk.vtkPolyData()
wedge_poly = make_polydata(wedge_pts) if len(wedge_pts) else vtk.vtkPolyData()

# --- ParaView pipeline -----------------------------------------------------

field = OpenDataFile(INPUT_FILE)
RenameSource("brain", field)
fieldDisplay = Show(field, GetActiveViewOrCreate('RenderView'))
ColorBy(fieldDisplay, ('POINTS', 'A'))
fieldDisplay.SetScalarBarVisibility(GetActiveViewOrCreate('RenderView'), True)
fieldDisplay.SetRepresentationType('Surface')

trisectorProducer = TrivialProducer(registrationName="Trisectors")
trisectorProducer.GetClientSideObject().SetOutput(trisector_poly)

wedgeProducer = TrivialProducer(registrationName="Wedges")
wedgeProducer.GetClientSideObject().SetOutput(wedge_poly)

view = GetActiveViewOrCreate('RenderView')

trisectorGlyph = Glyph(Input=trisectorProducer, GlyphType='Sphere')
trisectorGlyph.GlyphType.Radius = 1.0
trisectorGlyph.ScaleFactor = 1.0
trisectorGlyph.GlyphMode = 'All Points'
trisectorDisplay = Show(trisectorGlyph, view)
trisectorDisplay.DiffuseColor = [1.0, 0.4118, 0.7059]  # pink
trisectorDisplay.AmbientColor = [1.0, 0.4118, 0.7059]

wedgeGlyph = Glyph(Input=wedgeProducer, GlyphType='Sphere')
wedgeGlyph.GlyphType.Radius = 1.0
wedgeGlyph.ScaleFactor = 1.0
wedgeGlyph.GlyphMode = 'All Points'
wedgeDisplay = Show(wedgeGlyph, view)
wedgeDisplay.DiffuseColor = [1.0, 1.0, 1.0]  # white
wedgeDisplay.AmbientColor = [1.0, 1.0, 1.0]

view.InteractionMode = '2D'
view.OrientationAxesVisibility = 0
ResetCamera(view)
Render(view)

SaveScreenshot("./brain_degenerate_points.png", view, ImageResolution=[1600, 1000])
print("Saved screenshot to ./brain_degenerate_points.png")
