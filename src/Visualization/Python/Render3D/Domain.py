# Distributed under the MIT License.
# See LICENSE.txt for details.

from typing import Optional

import click
import numpy as np
import rich


def _parse_step(ctx, param, value):
    if value is None:
        return None
    if value.lower() == "first":
        return 0
    if value.lower() == "last":
        return -1
    return int(value)


def render_domain(
    xmf_file: str,
    output: str,
    hi_res_xmf_file: Optional[str] = None,
    time_step: int = -1,
    animate: bool = False,
    zoom_factor: float = 1.0,
    camera_theta: float = 0.0,
    camera_phi: float = 0.0,
    clip_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    clip_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    slice: bool = False,
):
    import paraview.simple as pv

    # Load data
    volume_data = pv.XDMFReader(
        registrationName="VolumeData", FileNames=[xmf_file]
    )
    if hi_res_xmf_file:
        hi_res_volume_data = pv.XDMFReader(
            registrationName="HiResVolumeData", FileNames=[hi_res_xmf_file]
        )

    render_view = pv.GetActiveViewOrCreate("RenderView")
    render_view.UseLight = 0
    render_view.UseColorPaletteForBackground = 0
    render_view.Background = 3 * [1.0]
    render_view.OrientationAxesVisibility = 0

    def slice_or_clip(triangulate, **kwargs):
        if slice:
            result = pv.Slice(
                **kwargs,
                SliceType="Plane",
                Triangulatetheslice=triangulate,
            )
            result.SliceType.Origin = clip_origin
            result.SliceType.Normal = clip_normal
        else:
            result = pv.Clip(**kwargs, ClipType="Plane")
            result.ClipType.Origin = clip_origin
            result.ClipType.Normal = clip_normal
        return result

    # Show grid
    grid = slice_or_clip(
        registrationName="Grid", Input=volume_data, triangulate=False
    )
    grid_display = pv.Show(grid, render_view)
    grid_display.Representation = "Surface With Edges"
    grid_display.LineWidth = 1.0
    grid_display.EdgeColor = 3 * [0.6]
    # The following line works around a failure in `pv.ColorBy(..., None)`
    grid_display.ColorArrayName = ("POINTS", None)
    pv.ColorBy(grid_display, None)

    # Show outline
    outline_data = hi_res_volume_data if hi_res_xmf_file else volume_data
    outline = slice_or_clip(
        registrationName="Outline", Input=outline_data, triangulate=True
    )
    outline_display = pv.Show(outline, render_view)
    outline_display.Representation = "Feature Edges"
    outline_display.LineWidth = 3.0
    outline_display.DiffuseColor = 3 * [0.0]
    outline_display.AmbientColor = 3 * [0.0]
    # The following line works around a failure in `pv.ColorBy(..., None)`
    outline_display.ColorArrayName = ("POINTS", None)
    pv.ColorBy(outline_display, None)

    # Set resolution
    layout = pv.GetLayout()
    layout.SetSize(1200, 1200)

    # Configure camera
    camera = pv.GetActiveCamera()
    camera.SetFocalPoint(*clip_origin)
    camera.SetPosition(*(np.array(clip_origin) + np.array(clip_normal) * 10.0))
    camera.Roll(camera_phi)
    camera.Elevation(-camera_theta)
    pv.ResetCamera()
    camera.Zoom(zoom_factor)

    if animate:
        pv.SaveAnimation(output, render_view)
    else:
        render_view.ViewTime = volume_data.TimestepValues[time_step]
        pv.Render()
        pv.SaveScreenshot(output, render_view)


@click.command(name="domain")
@click.argument(
    "xmf_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.argument(
    "hi_res_xmf_file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    required=False,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(writable=True),
    required=True,
    help="Output file. Include extension such as '.png'.",
)
@click.option(
    "--time-step",
    "-t",
    callback=_parse_step,
    default="first",
    show_default=True,
    help=(
        "Select a time step. Specify '-1' or 'last' to select the last time"
        " step."
    ),
)
@click.option(
    "--animate", is_flag=True, help="Produce an animation of all time steps."
)
@click.option("zoom_factor", "--zoom", help="Zoom factor.", default=1.0)
@click.option(
    "--camera-theta",
    type=float,
    default=0.0,
    help="Viewing angle from the z-axis in degrees.",
    show_default=True,
)
@click.option(
    "--camera-phi",
    type=float,
    default=0.0,
    help="Viewing angle around the z-axis in degrees.",
    show_default=True,
)
@click.option(
    "--clip-origin",
    "--slice-origin",
    nargs=3,
    type=float,
    default=(0.0, 0.0, 0.0),
    help="Origin of the clipping plane",
    show_default=True,
)
@click.option(
    "--clip-normal",
    "--slice-normal",
    nargs=3,
    type=float,
    default=(0.0, 0.0, 1.0),
    help="Normal of the clipping plane",
    show_default=True,
)
@click.option(
    "--slice/--clip",
    "slice",
    default=False,
    help="Use a slice instead of a clip.",
    show_default=True,
)
def render_domain_command(**kwargs):
    """Renders a 3D domain with elements and grid lines

    This rendering is a starting point for visualizations of the domain
    geometry, e.g. for publications.

    XMF_FILE is the path to the XMF file that references the simulation data.
    It is typically generated by the 'generate-xdmf' command.
    You can also provide a second XMF file with higher resolution data, which
    is used to render the outlines of elements to make them smoother.
    """
    render_domain(**kwargs)


if __name__ == "__main__":
    render_domain_command(help_option_names=["-h", "--help"])
