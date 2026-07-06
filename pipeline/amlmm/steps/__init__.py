"""Importing this package self-registers every step into amlmm.step.REGISTRY."""
from . import feasibility          # noqa: F401
from . import assemble_features    # noqa: F401
from . import classify             # noqa: F401
from . import cluster_explore      # noqa: F401
from . import report               # noqa: F401
from . import discover             # noqa: F401  (three-agent rebuild: the Discovery agent)

DEFAULT_ORDER = ["feasibility", "assemble_features", "classify", "cluster_explore", "report"]
# the Discovery agent runs as its own order (NOT folded into the baseline DEFAULT_ORDER)
DISCOVERY_ORDER = ["discover"]
