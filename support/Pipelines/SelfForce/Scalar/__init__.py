# Distributed under the MIT License.
# See LICENSE.txt for details.

import click

from spectre.support.CliExceptions import RequiredChoiceError


# Load subcommands lazily, i.e., only import the module when the subcommand is
# invoked. This is important so the CLI responds quickly.
class ScalarSelfForce(click.MultiCommand):
    def list_commands(self, ctx):
        return [
            "assemble",
            "solve",
        ]

    def get_command(self, ctx, name):
        if name == "assemble":
            from .Assemble import assemble_scalar_self_force_command

            return assemble_scalar_self_force_command
        elif name == "solve":
            from .Solve import solve_scalar_self_force_command

            return solve_scalar_self_force_command
        raise RequiredChoiceError(
            f"The command '{name}' is not implemented.",
            choices=self.list_commands(ctx),
        )


@click.group(name="scalar-self-force", cls=ScalarSelfForce)
def scalar_self_force_pipeline():
    """Pipeline for scalar self force simulations."""
    pass


if __name__ == "__main__":
    scalar_self_force_pipeline(help_option_names=["-h", "--help"])
