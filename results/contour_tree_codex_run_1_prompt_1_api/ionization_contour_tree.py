from paraview.simple import *
from topopilot.nodes import *
import topopilot.config as config
import topopilot.nodeTree as tree
import argparse
import os
from pathlib import Path

DEFAULT_RANDOM_SEED = 0


def pipeline(load_path):
    # Keep the original data as the first rendered branch.
    scalar = LoadScalarField(load_path=load_path, arrayName="Scalars_")
    scalar_view = EmbedScalarField(parent_node_id=scalar.id, colormap="viridis")

    # Compute the requested piecewise-linear contour tree after persistence
    # simplification of 10% of the scalar range.
    simplified = PersistenceSimplification(parent_node_id=scalar.id, epsilon=0.1)
    contour_tree = ComputeContourTree(parent_node_id=simplified.id)
    contour_tree_view = EmbedContourTree(
        parent_node_id=contour_tree.id,
        ball_radius=2.0,
        tube_radius=1.0,
        # minima, 1-saddles, 2-saddles, maxima, degenerate points
        color_scheme=["#0000ff", "#ffffff", "#ffa500", "#ff0000", "#000000"],
    )

    return [scalar_view, contour_tree_view], {}, [], []


def create_view():
    view = CreateView("RenderView")
    view.ViewSize = [1280, 720]
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.CenterOfRotation = [149.5, 61.5, 61.5]
    view.CameraPosition = [450.0, 360.0, 420.0]
    view.CameraFocalPoint = [149.5, 61.5, 61.5]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = True
    view.CameraParallelScale = 250.0
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
                print(f"missing file {dirname}/{basename}/{basename}{i}.vti")
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
        print("error: when outputting to video format your output filename must end in .mp4")
        exit(1)
    if args.t != -1 and not args.x and args.o[-4:] != ".png":
        print("error: when outputting to image format your output filename must end in .png")
        exit(1)
    if args.t == -1 and time_steps == 1:
        args.t = 1
    return time_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ionization scalar field and simplified contour-tree renderer")
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
    session_dir = Path(config.session_folder())
    session_dir.mkdir(exist_ok=True)
    (session_dir / "cache").mkdir(exist_ok=True)

    nodes_to_visualize, color_bar_info, _, _ = pipeline(args.i)

    if args.t != -1:
        render_view = create_view()
        for node in nodes_to_visualize:
            node.visualize(args.t, render_view)
        if args.o:
            SaveScreenshot(args.o, render_view)
        else:
            Interact()
    else:
        for t in range(time_steps):
            render_view = create_view()
            for node in nodes_to_visualize:
                node.visualize(t + 1, render_view)
            SaveScreenshot(f"{args.q}/frame{str(t + 1).zfill(8)}.png", render_view)
            Disconnect()
            tree.clear_evaluation_cache()
            Connect()

        os.system(f"ffmpeg -y -framerate 4 -i {args.q}/frame%08d.png -c:v libx264 -pix_fmt yuv420p {args.o}")
        for t in range(time_steps):
            os.remove(f"{args.q}/frame{str(t + 1).zfill(8)}.png")
