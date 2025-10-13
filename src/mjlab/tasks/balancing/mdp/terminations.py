"""Termination terms for single-leg balancing task."""

import torch

from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def undesired_body_ground_contact(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "undesired_body_ground_contact",
) -> torch.Tensor:
  """Terminate if any non-foot body part touches the ground.
  
  This is the PRIMARY fall detection method. Uses a contact sensor that monitors
  the pelvis subtree (which includes hands, knees, torso, etc.) for ground contact.
  Feet/ankles are excluded from this sensor, so only undesired contacts are detected.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    sensor_name: Name of the contact sensor monitoring unwanted ground contacts.
    
  Returns:
    Tensor of shape (num_envs,) with True if fallen, False otherwise.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Check the contact sensor for non-foot body ground contact
  # Sensor returns > 0 if any contact is detected
  contact_data = asset.data.sensor_data[sensor_name]  # Shape: (num_envs, num_contacts)
  
  # If ANY contact is detected, robot has fallen
  has_contact = contact_data[:, 0] > 0
  
  return has_contact.bool()


def raised_foot_contact(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate episode if the raised foot (the one that should be off the ground) touches the ground.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs,) with True if raised foot touches ground, False otherwise.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get standing leg choice from extras (0 = left, 1 = right)
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0
  
  # Check if the raised foot (opposite of standing leg) is in contact
  # standing_leg = 0 (left) -> check right foot contact
  # standing_leg = 1 (right) -> check left foot contact
  raised_foot_in_contact = torch.where(
    standing_leg == 0,
    right_foot_contact,
    left_foot_contact,
  )
  
  return raised_foot_in_contact.bool()


def root_height_below_minimum(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  minimum_height: float = 0.5,
) -> torch.Tensor:
  """Terminate episode if robot's root (torso) height falls below a minimum threshold.
  
  This catches cases where the robot collapses or crouches too low.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    minimum_height: Minimum allowed root height in meters (default: 0.5m).
    
  Returns:
    Tensor of shape (num_envs,) with True if root is too low, False otherwise.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get root body position
  root_pose = asset.data.root_link_pose_w  # Shape: (num_envs, 7) [x, y, z, quat]
  root_height = root_pose[:, 2]  # Z-coordinate
  
  # Check if root height is below minimum
  too_low = root_height < minimum_height
  
  return too_low.bool()


def both_feet_off_ground(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate episode if both feet are off the ground (jumping/hopping).
  
  This prevents the robot from "cheating" by jumping to get foot clearance rewards.
  The task is single-leg STANDING, not jumping.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs,) with True if both feet are off ground, False otherwise.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0
  
  # Both feet off ground = neither foot in contact
  both_off = ~left_foot_contact & ~right_foot_contact
  
  return both_off.bool()
