"""100K+ active-league control plane.

The package is isolated from the running 20K control.  Its initial mode is
strictly shadow-only: it may measure and propose decisions, but it cannot
change matchmaking, promote a champion, run a side learner, or launch a GPU
trainer until the migration and staged-release gates have been satisfied.
"""

from .contracts import VNextConfig, VNextMode, VNextStage
from .shadow import VNextShadowController

VNEXT_PROTOCOL = "active_league_100k_vnext_control_v1"

__all__ = [
    "VNEXT_PROTOCOL",
    "VNextConfig",
    "VNextMode",
    "VNextStage",
    "VNextShadowController",
]
