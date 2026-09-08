from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    # Keep the unmodified source visible, while deriving the topology from its
    # persistence-simplified counterpart.
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    raw_field_view = EmbedScalarField(parent_node_id=scalar.id, colormap="viridis")

    simplified = PersistenceSimplification(parent_node_id=scalar.id, epsilon=0.1)
    contour_tree = ComputeContourTree(parent_node_id=simplified.id)
    contour_tree_view = EmbedContourTree(
        parent_node_id=contour_tree.id,
        tube_radius=1.0,
        ball_radius=2.0,
        # minima, 1-saddles, 2-saddles, maxima, degenerate points
        color_scheme=["#0000ff", "#ffffff", "#ffa500", "#ff0000", "#000000"],
    )

    return [raw_field_view, contour_tree_view], {
        raw_field_view.id: ([0.82, 0.15], 0.70, "Scalars_"),
    }, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 720]
    view.OrientationAxesVisibility = 1
    view.CenterOfRotation = [149.5, 61.5, 61.5]
    view.CameraPosition = [550.0, 450.0, 500.0]
    view.CameraFocalPoint = [149.5, 61.5, 61.5]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 330.0
    view.InteractionMode = "3D"
    return view


def validate_arguments(args):
    if args.i[-4:] == ".vti":
        time_steps = 1
    else:
        if not os.path.isdir(args.i):
            print("error: file specified with -i must either be .vti file or directory")
            exit(1)
        time_steps = len(os.listdir(args.i))
        basename = os.path.basename(args.i)
        dirname = os.path.dirname(args.i)
        for i in range(1, time_steps + 1):
            if not os.path.isfile(f"{dirname}/{basename}/{basename}{i}.vti"):
                print("error: invalid input format for directory")
                exit(1)
    if args.t > time_steps:
        print("error: time step is too high")
        exit(1)
    if args.x and args.t == -1 and time_steps > 1:
        print("error: cannot run interactive mode on multiple time steps")
        exit(1)
    if args.o != "" and args.x:
        print("error: you cannot specify an output file and -interactive")
        exit(1)
    if args.o == "" and not args.x:
        print("error: you must either specify an output file or run in interactive mode.")
        exit(1)
    if args.t == -1 and time_steps > 1 and args.o[-4:] != ".mp4":
        print("error: when outputting all time steps your output filename must end in .mp4")
        exit(1)
    if args.t != -1 and not args.x and args.o[-4:] != ".png":
        print("error: when outputting one time step your output filename must end in .png")
        exit(1)
    if args.t == -1 and time_steps == 1:
        args.t = 1
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True)
    parser.add_argument("-x", action=argparse.BooleanOptionalAction)
    parser.add_argument("-o", default="")
    parser.add_argument("-t", type=int, default=-1)
    parser.add_argument("-q", default=".")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    time_steps = validate_arguments(args)
    config.set_random_seed(args.seed)
    config.change_session_folder(args.q)
    Path(config.session_folder()).mkdir(exist_ok=True)
    Path(f"{config.session_folder()}/cache").mkdir(exist_ok=True)
    nodes, color_bars, _, _ = pipeline(args.i)
    if args.t != -1:
        view = create_view()
        view.Background = [1.0, 1.0, 1.0]
        view.UseColorPaletteForBackground = 0
        for node in nodes:
            props = node.visualize(args.t, view)
            if node.id in color_bars:
                position, length, title = color_bars[node.id]
                bar = GetScalarBar(props.LookupTable, view)
                bar.WindowLocation, bar.Position, bar.ScalarBarLength = "Any Location", position, length
                bar.Title, bar.ComponentTitle = title, ""
                props.SetScalarBarVisibility(view, True)
        if args.o:
            SaveScreenshot(args.o, view)
        else:
            Interact()
    else:
        for t in range(1, time_steps + 1):
            view = create_view()
            view.Background = [1.0, 1.0, 1.0]
            view.UseColorPaletteForBackground = 0
            for node in nodes:
                props = node.visualize(t, view)
                if node.id in color_bars:
                    position, length, title = color_bars[node.id]
                    bar = GetScalarBar(props.LookupTable, view)
                    bar.WindowLocation, bar.Position, bar.ScalarBarLength = "Any Location", position, length
                    bar.Title, bar.ComponentTitle = title, ""
                    props.SetScalarBarVisibility(view, True)
            SaveScreenshot(f"{args.q}/frame{str(t).zfill(8)}.png", view)
            Disconnect(); tree.clear_evaluation_cache(); Connect()
        os.system(f"ffmpeg -y -framerate 4 -i {args.q}/frame%08d.png -c:v libx264 -pix_fmt yuv420p {args.o}")
        for t in range(1, time_steps + 1):
            os.remove(f"{args.q}/frame{str(t).zfill(8)}.png")
