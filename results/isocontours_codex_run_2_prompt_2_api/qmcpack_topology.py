from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path


def pipeline(load_path):
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    simplified = PersistenceSimplification(
        parent_node_id=scalar.id,
        epsilon=0.04,
    )

    # Four quantile isovalues partition the scalar range into five equally sized
    # (by sampled volume) regions.
    contours = ComputeIsocontours(
        parent_node_id=simplified.id,
        num_contours=4,
    )
    contour_view = EmbedIsocontours(
        parent_node_id=contours.id,
        colormap="viridis",
    )

    critical_points = ComputeScalarCriticalPoints(
        parent_node_id=simplified.id,
        which_points="all critical points (mins+maxes+saddles)",
    )
    critical_point_view = EmbedScalarCriticalPoints(
        parent_node_id=critical_points.id,
        radius=1.5,
        # minima, 1-saddles, 2-saddles, maxima, degenerate points
        color_scheme=["#0000ff", "#ffffff", "#ffa500", "#ff0000", "#000000"],
    )

    return [contour_view, critical_point_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 900]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [34.0, 34.0, 57.0]
    view.CameraPosition = [155.0, 145.0, 175.0]
    view.CameraFocalPoint = [34.0, 34.0, 57.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 105.0
    view.InteractionMode = "3D"
    return view


def validate_arguments(args):
    if args.i[-4:] != ".vti":
        raise ValueError("This visualization expects a single .vti file")
    if args.t not in (-1, 1):
        raise ValueError("A single VTI supports only time step 1")
    if args.o == "" and not args.x:
        raise ValueError("Specify -o OUTPUT.png or use -x")
    if args.o and args.x:
        raise ValueError("-o and -x cannot be combined")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", default="")
    parser.add_argument("-t", type=int, default=-1)
    parser.add_argument("-x", action=argparse.BooleanOptionalAction)
    parser.add_argument("-q", default=".")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    validate_arguments(args)

    config.set_random_seed(args.seed)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)

    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)
    render_view = create_view()
    render_view.Background = [0.08, 0.08, 0.08]
    render_view.UseColorPaletteForBackground = 0

    for node in nodes_to_visualize:
        node.visualize(1, render_view)

    # Semi-transparent contour surfaces keep interior critical points visible.
    for item in GetRepresentations():
        # ParaView 5.13 returns (name, proxy) pairs here.
        representation = item[-1] if isinstance(item, tuple) else item
        if hasattr(representation, "ColorArrayName") and representation.ColorArrayName == ["POINTS", "Scalars_"]:
            representation.Opacity = 0.30

    if args.o:
        SaveScreenshot(args.o, render_view)
    else:
        Interact()
