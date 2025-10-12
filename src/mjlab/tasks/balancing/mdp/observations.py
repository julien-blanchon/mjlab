"""Observation terms for single-leg balancing task."""

import torch

from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def standing_leg_indicator(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Returns a one-hot vector indicating which leg is the standing leg.
  
  Returns [1, 0] if left leg is standing, [0, 1] if right leg is standing.
  The standing leg choice is stored in env.extras at episode reset.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration (unused, for API consistency).
    
  Returns:
    Tensor of shape (num_envs, 2) with one-hot encoding.
  """
  # Get standing leg choice from extras (0 = left, 1 = right)
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Create one-hot encoding
  one_hot = torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)
  one_hot[torch.arange(env.num_envs, device=env.device), standing_leg] = 1.0
  
  return one_hot


def raised_knee_height(
  env: ManagerBasedEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Returns the z-position (height) of the raised knee above ground.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs, 1) with knee height in meters.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get standing leg choice from extras (0 = left, 1 = right)
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get body IDs for left and right knee (override in robot-specific config)
  left_knee_ids, _ = asset.find_bodies(["left_knee_link"])
  right_knee_ids, _ = asset.find_bodies(["right_knee_link"])
  
  # Get knee positions in world frame
  left_knee_pos = asset.data.body_link_pos_w[:, left_knee_ids[0], 2:3]  # z-coordinate
  right_knee_pos = asset.data.body_link_pos_w[:, right_knee_ids[0], 2:3]  # z-coordinate
  
  # Select the raised knee (opposite of standing leg)
  # standing_leg = 0 (left) -> raised = right (1)
  # standing_leg = 1 (right) -> raised = left (0)
  raised_knee_pos = torch.where(
    standing_leg.unsqueeze(1) == 0,
    right_knee_pos,
    left_knee_pos,
  )
  
  return raised_knee_pos

