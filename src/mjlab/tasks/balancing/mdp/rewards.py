"""Reward terms for single-leg balancing task."""

import torch

from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for staying alive (maintaining balance).
  
  Returns a constant reward of 1.0 for each step the agent maintains balance.
  
  Args:
    env: The environment instance.
    
  Returns:
    Tensor of shape (num_envs,) with constant value 1.0.
  """
  return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


def knee_height_above_threshold(
  env: ManagerBasedRlEnv,
  threshold: float,
  std: float,
) -> torch.Tensor:
  """Reward for keeping the raised knee above a threshold height.
  
  Uses an exponential reward that increases as the knee gets higher than the threshold.
  
  Args:
    env: The environment instance.
    threshold: Minimum desired knee height in meters.
    std: Standard deviation for the exponential reward.
    
  Returns:
    Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene["robot"]
  
  # Get standing leg choice from extras (0 = left, 1 = right)
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get body IDs for left and right knee
  left_knee_ids, _ = asset.find_bodies(["left_knee_link"])
  right_knee_ids, _ = asset.find_bodies(["right_knee_link"])
  
  # Get knee heights (z-coordinate)
  left_knee_height = asset.data.body_link_pos_w[:, left_knee_ids[0], 2]
  right_knee_height = asset.data.body_link_pos_w[:, right_knee_ids[0], 2]
  
  # Select the raised knee height
  raised_knee_height = torch.where(
    standing_leg == 0,
    right_knee_height,
    left_knee_height,
  )
  
  # Calculate error from threshold (negative if below threshold)
  error = torch.clamp(raised_knee_height - threshold, min=-0.5, max=0.5)
  
  # Exponential reward: high reward when above threshold, low when below
  return torch.exp(error / std**2)


def upright_posture(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for maintaining an upright torso orientation.
  
  Rewards when projected gravity is close to [0, 0, -1] (upright).
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  
  # Ideal projected gravity is [0, 0, -1] when upright
  # Reward based on z-component being close to -1
  upright_error = torch.abs(projected_gravity[:, 2] + 1.0)
  
  return torch.exp(-upright_error / 0.25)


def base_stability(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for excessive base angular velocity.
  
  Encourages the robot to minimize rotational movements.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs,) with penalty values (positive, to be negated by weight).
  """
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.root_link_ang_vel_b
  
  # L2 norm of angular velocity
  return torch.sum(torch.square(ang_vel), dim=1)


class joint_posture:
  """Reward for maintaining joint positions close to default (standing posture).
  
  This is implemented as a class to pre-compute the standard deviations per joint.
  Note: We could optionally exclude raised leg joints from this reward.
  """
  
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos
    
    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    
    _, _, std = resolve_matching_names_values(
      data=cfg.params["std"],
      list_of_strings=joint_names,
    )
    self.std = torch.tensor(std, device=env.device, dtype=torch.float32)
  
  def __call__(
    self, env: ManagerBasedRlEnv, std, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    del std  # Unused, we use pre-computed std from __init__
    asset: Entity = env.scene[asset_cfg.name]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)
    return torch.exp(-torch.mean(error_squared / (self.std**2), dim=1))

