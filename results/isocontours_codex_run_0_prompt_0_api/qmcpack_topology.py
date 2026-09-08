from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import argparse
from pathlib import Path


def pipeline(load_path):
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    simplified = PersistenceSimplification(parent_node_id=scalar.id, epsilon=0.04)

    field_view = EmbedScalarField(parent_node_id=simplified.id, colormap="viridis")
    contours = ComputeIsocontours(parent_node_id=simplified.id, num_contours=4)
    contour_view = EmbedIsocontours(
        parent_node_id=contours.id, colormap="viridis", tube_radius=1.0
    )
    critical_points = ComputeScalarCriticalPoints(
        parent_node_id=simplified.id,
        which_points="all critical points (mins+maxes+saddles)",
    )
    critical_points_view = EmbedScalarCriticalPoints(
        parent_node_id=critical_points.id,
        radius=2.0,
        # minima, 1-saddles, 2-saddles, maxima, degenerate points
        color_scheme=["#0000ff", "#ffffff", "#ffa500", "#ff0000", "#000000"],
    )

    return [field_view, contour_view, critical_points_view], {
        field_view.id: ([0.84, 0.14], 0.70, "Scalars_ (simplified)"),
    }, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 900]
    view.OrientationAxesVisibility = 1
    view.CameraPosition = [150.0, 150.0, 190.0]
    view.CameraFocalPoint = [34.0, 34.0, 57.0]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 120.0
    view.InteractionMode = "3D"
    return view


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-o", required=True)
    parser.add_argument("-t", type=int, default=1)
    parser.add_argument("-q", default=".")
    args = parser.parse_args()

    config.set_random_seed(0)
    config.change_session_folder(args.q)
    Path(args.q).mkdir(exist_ok=True)
    Path(args.q, "cache").mkdir(exist_ok=True)
    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)

    render_view = create_view()
    render_view.Background = [1.0, 1.0, 1.0]
    render_view.UseColorPaletteForBackground = 0
    for node in nodes_to_visualize:
        props = node.visualize(args.t, render_view)
        if node.id in color_bar_info:
            position, length, title = color_bar_info[node.id]
            bar = GetScalarBar(props.LookupTable, render_view)
            bar.WindowLocation = "Any Location"
            bar.Position = position
            bar.ScalarBarLength = length
            bar.Title = title
            bar.ComponentTitle = ""
            props.SetScalarBarVisibility(render_view, True)
    SaveScreenshot(args.o, render_view)
