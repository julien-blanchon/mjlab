"""Termination terms for single-leg balancing task."""

import torch

from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def raised_foot_contact(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate if the raised foot makes contact with the ground.
  
  This termination ensures true single-leg balancing by ending the episode
  if the raised foot touches the ground.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Boolean tensor of shape (num_envs,) indicating termination.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get standing leg choice from extras (0 = left, 1 = right)
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0
  
  # Determine which foot is raised (opposite of standing)
  # If standing_leg == 0 (left), then raised is right (check right_foot_contact)
  # If standing_leg == 1 (right), then raised is left (check left_foot_contact)
  raised_foot_touched = torch.where(
    standing_leg == 0,
    right_foot_contact,
    left_foot_contact,
  )
  
  return raised_foot_touched

