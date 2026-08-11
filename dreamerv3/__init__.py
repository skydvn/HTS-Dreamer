"""
htp_wm — Hierarchical Temporal Prefix World Model as a DreamerV3 mod pack.

Drop this directory next to your existing DreamerV3 code and run
`python -m htp_wm.main --configs atari100k` (or any other config preset,
after merging the additions from `configs.yaml` into your own).

Public surface:

  Agent              (agent.Agent)          Drop-in replacement for DreamerV3's
                                            Agent. Setting config.htp.enabled=false
                                            reproduces the vanilla behaviour.

  OrderedProjection  (htp.OrderedProjection)      S_ψ : h -> z
  ProgressiveRecon   (htp.ProgressiveRecon)       Eq. 4  auxiliary loss
  MultiStridePDyn    (htp.MultiStridePDyn)        Eq. 7  auxiliary loss
"""

from . import htp
from . import rssm
from .agent_htp import Agent_HTP

__all__ = ['Agent_HTP', 'htp', 'rssm']