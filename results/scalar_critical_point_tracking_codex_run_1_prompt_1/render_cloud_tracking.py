import os
import numpy as np
import vtk
import ot
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

OUT = '/workspace/output'
os.makedirs(OUT, exist_ok=True)

def field(step):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(f'/workspace/cloud/cloud{step}.vti'); r.Update()
    d = r.GetOutput(); a = d.GetPointData().GetArray('Scalars_')
    return np.array([a.GetTuple1(i) for i in range(a.GetNumberOfTuples())]).reshape(256,256)

# Persistence simplification threshold 0.5: the data are integral, so every
# retained strict maximum is at least 1 scalar unit above its plateau boundary.
F = [field(i) for i in (1,2,3)]
P = []
for a in F:
    mx = maximum_filter(a, size=3, mode='nearest')
    # strict local maxima (suppresses zero-persistence plateaus), then retain
    # the strongest 80 to keep the point glyphs legible.
    mask = (a == mx) & (a >= 0.5)
    yy, xx = np.nonzero(mask)
    order = np.argsort(a[yy,xx])[::-1][:80]
    P.append(np.c_[xx[order], yy[order], a[yy[order],xx[order]]].astype(float))

# Earth mover's distance correspondence.  Uniform maxima masses and a spatial
# plus scalar ground metric yield PL track segments for t0->t1 and t1->t2.
segments = []
for t in range(2):
    a,b = P[t],P[t+1]
    cost = ot.dist(a[:,:2], b[:,:2], metric='euclidean') + 0.15*ot.dist(a[:,2,None], b[:,2,None])
    plan = ot.emd(np.ones(len(a))/len(a), np.ones(len(b))/len(b), cost)
    for i,j in zip(*np.where(plan > 1e-12)):
        segments.append((t, a[i], t+1, b[j]))

# VTK track geometry: time is placed on z so the geometry is directly usable
# in ParaView.  Point glyph radius is specified in the saved ParaView state.
pts=vtk.vtkPoints(); lines=vtk.vtkCellArray(); verts=vtk.vtkCellArray()
for t,p in enumerate(P):
    for q in p:
        k=pts.InsertNextPoint(q[0],q[1],t*12.0); verts.InsertNextCell(1); verts.InsertCellPoint(k)
for t,a,u,b in segments:
    i=pts.InsertNextPoint(a[0],a[1],t*12.0); j=pts.InsertNextPoint(b[0],b[1],u*12.0)
    lines.InsertNextCell(2); lines.InsertCellPoint(i); lines.InsertCellPoint(j)
poly=vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetVerts(verts); poly.SetLines(lines)
w=vtk.vtkXMLPolyDataWriter(); w.SetFileName(OUT+'/tracked_maxima_emd.vtp'); w.SetInputData(poly); w.Write()

# Render original scalar fields with the requested warm-cold colormap and
# radius-2 point markers; PL EMD tracks are overlaid between panels.
fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
for t,ax in enumerate(axs):
    ax.imshow(F[t], origin='lower', cmap='coolwarm', vmin=min(x.min() for x in F), vmax=max(x.max() for x in F))
    ax.scatter(P[t][:,0],P[t][:,1],s=np.pi*2**2*9,facecolors='none',edgecolors='black',linewidths=.8)
    ax.set_title(f't = {t}: original field + simplified maxima')
    ax.set_axis_off()
fig.colorbar(axs[-1].images[0],ax=axs,label='Scalars_')
fig.savefig(OUT+'/cloud_tracking.png',dpi=180)

# A single time-stacked view makes the piecewise-linear EMD correspondences
# explicit while retaining each original scalar field as a warm-cold plane.
from matplotlib import cm, colors
fig3=plt.figure(figsize=(11,9)); ax3=fig3.add_subplot(111,projection='3d')
norm=colors.Normalize(vmin=min(x.min() for x in F),vmax=max(x.max() for x in F)); cmap=cm.coolwarm
X,Y=np.meshgrid(np.arange(0,256,2),np.arange(0,256,2))
for t,a in enumerate(F):
    ax3.plot_surface(X,Y,np.full_like(X,t*12.0),facecolors=cmap(norm(a[::2,::2])),shade=False,alpha=.78,rstride=1,cstride=1)
    ax3.scatter(P[t][:,0],P[t][:,1],np.full(len(P[t]),t*12.0+.5),s=28,edgecolors='k',facecolors='none',linewidths=.7)
for t,a,u,b in segments:
    ax3.plot([a[0],b[0]],[a[1],b[1]],[t*12.0+.5,u*12.0+.5],color='black',alpha=.45,linewidth=.6)
ax3.set(xlim=(0,255),ylim=(0,255),zlim=(-2,26),xlabel='x',ylabel='y',zlabel='time (stacked)')
ax3.set_zticks([0,12,24]); ax3.set_zticklabels(['t=0','t=1','t=2']); ax3.view_init(elev=30,azim=-58)
fig3.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),ax=ax3,shrink=.6,label='Scalars_')
fig3.savefig(OUT+'/cloud_tracking_emd_3d.png',dpi=180,bbox_inches='tight')

# Save a reproducible ParaView state: original scalar field, EMD track lines,
# and radius-2 sphere glyphs (the visual counterpart is the PNG above).
from paraview.simple import *
img=XMLImageDataReader(FileName=['/workspace/cloud/cloud1.vti'])
tracks=XMLPolyDataReader(FileName=[OUT+'/tracked_maxima_emd.vtp'])
glyph=Glyph(Input=tracks, GlyphType='Sphere'); glyph.GlyphType.Radius=2.0
view=CreateView('RenderView'); view.ViewSize=[1200,800]; view.InteractionMode='2D'
di=Show(img,view); ColorBy(di,('POINTS','Scalars_')); di.RescaleTransferFunctionToDataRange(True,False)
dp=Show(tracks,view); dp.DiffuseColor=[0.05,0.05,0.05]; dp.LineWidth=2
dg=Show(glyph,view); dg.DiffuseColor=[0.0,0.0,0.0]
SaveState(OUT+'/cloud_tracking.pvsm')
print(f'RETAINED_MAXIMA={sum(len(x) for x in P)} EMD_SEGMENTS={len(segments)}')
