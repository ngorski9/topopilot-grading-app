# Cylinder critical-point tracking

`track_cylinder.py` imports `cylinder1.vti`, `cylinder2.vti`, and
`cylinder3.vti`.  It locates common zeros of the piecewise-bilinear `u` and
`v` field within every image cell, then links consecutive time steps using
Python Optimal Transport's partial Wasserstein transport (85% mass).

Outputs:

- `cylinder_partial_ot_tracking.png` — original step-1 vector field (speed
  background plus arrows), all detected critical points (red/green/blue for
  steps 1/2/3), and black 3-step partial-OT tracks.
- `tracked_critical_points.vtp` — the point set for ParaView.
- `partial_ot_tracks.vtp` — polyline trajectories for ParaView.
- `critical_points_and_tracks.csv` — point-to-track table.

Rerun with `/opt/conda/bin/pvpython track_cylinder.py` from `/workspace`.
