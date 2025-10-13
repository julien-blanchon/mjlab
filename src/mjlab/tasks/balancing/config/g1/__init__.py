"""Gym environment registration for G1 single-leg balancing task."""

import gymnasium as gym

gym.register(
  id="Mjlab-Balancing-Flat-Unitree-G1",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1BalancingEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1BalancingPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Balancing-Flat-Unitree-G1-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1BalancingEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:G1BalancingPPORunnerCfg",
  },
)

