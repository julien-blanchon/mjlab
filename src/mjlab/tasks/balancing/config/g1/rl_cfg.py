"""RL configuration for G1 single-leg balancing task.

Reuses the PPO configuration from the G1 tracking task.
"""

from mjlab.tasks.tracking.config.g1.rl_cfg import G1FlatPPORunnerCfg

__all__ = ["G1FlatPPORunnerCfg"]

