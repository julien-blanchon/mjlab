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


def foot_clearance(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for lifting the raised foot off the ground WHILE standing foot is on ground.
  
  CRITICAL: This reward is ONLY given if:
  1. Raised foot is OFF the ground
  2. Standing foot IS ON the ground
  
  This prevents the "jumping hack" where robot lifts both feet.
  
  Returns a progressive reward based on raised foot height above ground:
  - 0.0 if foot on ground or both feet off ground
  - 0.5-1.0 for foot progressively higher (encourages clear lift, not just barely off ground)
  
  Args:
    env: The environment instance.
    
  Returns:
    Tensor of shape (num_envs,) with reward values (0 to 1).
  """
  asset: Entity = env.scene["robot"]
  
  # Get standing leg choice from extras (0 = left, 1 = right)
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0
  
  # Check if raised foot is off ground AND standing foot is on ground
  valid_single_leg = torch.where(
    standing_leg == 0,
    left_foot_contact & ~right_foot_contact,  # Left on, right off
    right_foot_contact & ~left_foot_contact,  # Right on, left off
  )
  
  # Get ankle/foot heights to measure how high the foot is lifted
  left_ankle_ids, _ = asset.find_bodies(["left_ankle_roll_link"])
  right_ankle_ids, _ = asset.find_bodies(["right_ankle_roll_link"])
  
  left_ankle_height = asset.data.body_link_pos_w[:, left_ankle_ids[0], 2]
  right_ankle_height = asset.data.body_link_pos_w[:, right_ankle_ids[0], 2]
  
  # Get raised foot height and standing foot height
  raised_foot_height = torch.where(
    standing_leg == 0,
    right_ankle_height,
    left_ankle_height,
  )
  standing_foot_height = torch.where(
    standing_leg == 0,
    left_ankle_height,
    right_ankle_height,
  )
  
  # Calculate how high the raised foot is above standing foot
  foot_height_above_ground = raised_foot_height - standing_foot_height
  
  # ADDITIONAL CHECK: Foot must be meaningfully lifted (>5cm)
  # This prevents getting reward from tiny height differences when both feet are on ground
  min_clearance = 0.05  # 5cm absolute minimum
  foot_actually_lifted = foot_height_above_ground > min_clearance
  
  # Progressive reward: 0.5 at 5cm lift, 1.0 at 15cm+ lift
  # This encourages CLEAR lifting, not just barely touching off ground
  height_reward = 0.5 + 0.5 * torch.clamp(foot_height_above_ground / 0.15, min=0.0, max=1.0)
  
  # Only give reward if BOTH: valid single-leg stance AND foot actually lifted >5cm
  both_conditions = valid_single_leg & foot_actually_lifted
  
  return torch.where(
    both_conditions,
    height_reward,
    torch.zeros_like(height_reward),
  )


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for staying alive (maintaining balance).
  
  Returns a constant reward of 1.0 for each step the agent maintains balance.
  
  Args:
    env: The environment instance.
    
  Returns:
    Tensor of shape (num_envs,) with constant value 1.0.
  """
  return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


def raised_foot_height_optimal_range(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_height: float = 0.15,
  optimal_height: float = 0.25,
  max_height: float = 0.35,
) -> torch.Tensor:
  """Reward for keeping raised FOOT in optimal height range (classical tree pose).
  
  CRITICAL: Measures FOOT/ankle height (not knee), so robot can't cheat by bending knee.
  CRITICAL: This reward is ONLY given when in valid single-leg stance.
  
  Reward structure:
  - 0-0.15m: Small reward (0 → 0.3)
  - 0.15-0.25m: Ramping to maximum (0.3 → 1.0)
  - 0.25-0.30m: Maximum reward (1.0) ← Classical tree pose range
  - 0.30-0.35m: Gentle decay (1.0 → 0.5)
  - >0.35m: PENALTY (too high!)
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    min_height: Minimum height to start getting rewards (0.15m).
    optimal_height: Target height for maximum reward (0.25m).
    max_height: Height above which penalties apply (0.35m).
    
  Returns:
    Tensor of shape (num_envs,) with reward values (-inf to 1.0).
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get standing leg choice
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0
  
  # Check if in valid single-leg stance (standing foot ON, raised foot OFF)
  valid_single_leg = torch.where(
    standing_leg == 0,
    left_foot_contact & ~right_foot_contact,  # Left on, right off
    right_foot_contact & ~left_foot_contact,  # Right on, left off
  )
  
  # Get ankle body indices (measure FOOT height, not knee)
  left_ankle_ids, _ = asset.find_bodies(["left_ankle_roll_link"])
  right_ankle_ids, _ = asset.find_bodies(["right_ankle_roll_link"])
  
  # Get ankle heights
  left_ankle_z = asset.data.body_link_pos_w[:, left_ankle_ids[0], 2]
  right_ankle_z = asset.data.body_link_pos_w[:, right_ankle_ids[0], 2]
  
  # Calculate raised FOOT height relative to standing foot
  standing_foot_z = torch.where(standing_leg == 0, left_ankle_z, right_ankle_z)
  raised_foot_z = torch.where(standing_leg == 0, right_ankle_z, left_ankle_z)
  foot_height = raised_foot_z - standing_foot_z
  
  # ADDITIONAL CHECK: Raised foot must be meaningfully above ground
  # This prevents getting reward from foot position differences when both are on ground
  min_clearance = 0.05  # 5cm absolute minimum clearance
  foot_actually_lifted = foot_height > min_clearance
  
  # Reward structure:
  reward = torch.zeros_like(foot_height)
  
  # Region 1: Below min_height (0 - 0.15m) - small reward for any lifting
  below_min = foot_height < min_height
  reward[below_min] = 0.3 * (foot_height[below_min] / min_height)
  
  # Region 2: Min to optimal (0.15 - 0.25m) - ramping to maximum
  in_ramp_up = (foot_height >= min_height) & (foot_height < optimal_height)
  ramp_up_progress = (foot_height[in_ramp_up] - min_height) / (optimal_height - min_height)
  reward[in_ramp_up] = 0.3 + 0.7 * ramp_up_progress  # 0.3 → 1.0
  
  # Region 3: Optimal range (0.25-0.30m) - maximum reward (classical tree pose)
  in_optimal = (foot_height >= optimal_height) & (foot_height <= optimal_height + 0.05)
  reward[in_optimal] = 1.0
  
  # Region 4: Approaching max (0.30-0.35m) - gentle decay
  in_decay = (foot_height > optimal_height + 0.05) & (foot_height <= max_height)
  decay_progress = (foot_height[in_decay] - (optimal_height + 0.05)) / (max_height - optimal_height - 0.05)
  reward[in_decay] = 1.0 - 0.7 * decay_progress  # 1.0 → 0.3
  
  # Region 5: Above max (>0.35m) - PENALTY ZONE (too high!)
  above_max = foot_height > max_height
  excess_height = foot_height[above_max] - max_height
  reward[above_max] = 0.3 - 3.0 * excess_height  # Strong penalty for excessive height
  
  # Only give reward if BOTH conditions met:
  # 1. Valid single-leg stance (standing foot ON, raised foot OFF per contact sensors)
  # 2. Foot actually lifted >5cm (prevents exploit from foot position differences)
  both_conditions = valid_single_leg & foot_actually_lifted
  
  return torch.where(
    both_conditions,
    reward,
    torch.zeros_like(reward),
  )


def knee_height_above_threshold(
  env: ManagerBasedRlEnv,
  threshold: float,
  std: float,
) -> torch.Tensor:
  """Reward for raising the knee above threshold height.
  
  CRITICAL: This reward is ONLY given when in valid single-leg stance:
  - Standing foot is ON the ground
  - Raised foot is OFF the ground
  
  This prevents the robot from getting reward by just sitting with both feet down
  (where knee is naturally elevated above ankle).
  
  Args:
    env: The environment instance.
    threshold: Target knee height in meters (0.4m).
    std: Standard deviation parameter (used for bonus curve steepness).
    
  Returns:
    Tensor of shape (num_envs,) with reward values (0 to ~1.5).
  """
  asset: Entity = env.scene["robot"]
  
  # Get standing leg choice from extras (0 = left, 1 = right)
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0
  
  # Check if in valid single-leg stance (standing foot ON, raised foot OFF)
  valid_single_leg = torch.where(
    standing_leg == 0,
    left_foot_contact & ~right_foot_contact,  # Left on, right off
    right_foot_contact & ~left_foot_contact,  # Right on, left off
  )
  
  # Get body IDs for left and right knee and ankle
  left_knee_ids, _ = asset.find_bodies(["left_knee_link"])
  right_knee_ids, _ = asset.find_bodies(["right_knee_link"])
  left_ankle_ids, _ = asset.find_bodies(["left_ankle_roll_link"])
  right_ankle_ids, _ = asset.find_bodies(["right_ankle_roll_link"])
  
  # Get knee heights (z-coordinate)
  left_knee_height = asset.data.body_link_pos_w[:, left_knee_ids[0], 2]
  right_knee_height = asset.data.body_link_pos_w[:, right_knee_ids[0], 2]
  
  # Get standing foot height (should be ~0, on ground)
  left_ankle_height = asset.data.body_link_pos_w[:, left_ankle_ids[0], 2]
  right_ankle_height = asset.data.body_link_pos_w[:, right_ankle_ids[0], 2]
  
  # Select the raised knee height and standing foot height
  raised_knee_height = torch.where(
    standing_leg == 0,
    right_knee_height,
    left_knee_height,
  )
  standing_foot_height = torch.where(
    standing_leg == 0,
    left_ankle_height,
    right_ankle_height,
  )
  
  # Calculate height relative to standing foot
  relative_knee_height = torch.clamp(raised_knee_height - standing_foot_height, min=0.0, max=1.0)
  
  # Progressive reward structure:
  # 1. Linear component: Rewards ANY height increase (0.0 → 1.0 as height goes 0 → threshold)
  linear_reward = torch.clamp(relative_knee_height / threshold, min=0.0, max=1.0)
  
  # 2. Bonus for reaching threshold: Extra reward when above threshold
  above_threshold = torch.clamp((relative_knee_height - threshold) / (threshold * 0.2), min=0.0, max=0.5)
  
  # Total reward ONLY if in valid single-leg stance
  # Otherwise 0 reward (prevents sitting exploit)
  total_reward = linear_reward + above_threshold
  
  return torch.where(
    valid_single_leg,
    total_reward,
    torch.zeros_like(total_reward),
  )


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


def weight_shift_preparation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for shifting weight onto the standing foot BEFORE lifting.
  
  This is the FIRST stage of learning - robot must learn to shift CoM
  over the standing foot. This makes lifting possible and safe.
  
  Key insight: The robot needs to unload the foot before it can lift it!
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get standing leg choice
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get both feet positions from contact sensor data
  left_foot_pos = asset.data.sensor_data["left_foot_ground_contact"][:, 1:4]  # XYZ position
  right_foot_pos = asset.data.sensor_data["right_foot_ground_contact"][:, 1:4]
  
  # Get root position (first 3 elements of root_link_pose_w are xyz position)
  root_pose = asset.data.root_link_pose_w  # Shape: (num_envs, 7) [x, y, z, qw, qx, qy, qz]
  com_pos = root_pose[:, :3]  # Extract xyz position
  
  # Calculate standing foot position
  standing_foot_pos = torch.where(
    standing_leg.unsqueeze(1) == 0,
    left_foot_pos,
    right_foot_pos,
  )
  
  # Calculate horizontal distance from CoM to standing foot
  com_to_foot_xy = com_pos[:, :2] - standing_foot_pos[:, :2]
  distance = torch.norm(com_to_foot_xy, dim=1)
  
  # Reward for CoM being close to standing foot (within 0.1m = excellent)
  # This teaches weight shift BEFORE attempting to lift
  reward = torch.exp(-distance / 0.05)  # Sharp falloff encourages tight alignment
  
  return reward


def conditional_upright_posture(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for upright posture ONLY when foot is off ground.
  
  This prevents the robot from getting upright reward while standing on both feet.
  Forces the robot to lift foot to get this reward.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs,) with reward values (0 if both feet on ground).
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Get standing leg choice from extras
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0
  
  # Check if raised foot is off ground
  raised_foot_off_ground = torch.where(
    standing_leg == 0,
    ~right_foot_contact,
    ~left_foot_contact,
  )
  
  # Calculate upright reward
  projected_gravity = asset.data.projected_gravity_b
  upright_error = torch.abs(projected_gravity[:, 2] + 1.0)
  upright_reward = torch.exp(-upright_error / 0.25)
  
  # Only give reward if foot is off ground
  return torch.where(
    raised_foot_off_ground,
    upright_reward,
    torch.zeros_like(upright_reward),
  )


def balanced_stance_duration(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for sustaining single-leg stance over time.
  
  Tracks how long the robot maintains a valid single-leg stance:
  - Raised foot OFF ground
  - Standing foot ON ground (prevents jumping hack)
  
  Counter resets if either condition is violated.
  
  Args:
    env: The environment instance.
    
  Returns:
    Tensor of shape (num_envs,) with reward values.
  """
  # Initialize counter if not present
  if "balanced_stance_counter" not in env.extras:
    env.extras["balanced_stance_counter"] = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
  
  asset: Entity = env.scene["robot"]
  
  # Get standing leg choice from extras
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get contact sensor data for both feet
  left_foot_contact = asset.data.sensor_data["left_foot_ground_contact"][:, 0] > 0
  right_foot_contact = asset.data.sensor_data["right_foot_ground_contact"][:, 0] > 0
  
  # Valid single-leg stance: standing foot ON, raised foot OFF
  in_balanced_stance = torch.where(
    standing_leg == 0,
    left_foot_contact & ~right_foot_contact,  # Left on, right off
    right_foot_contact & ~left_foot_contact,  # Right on, left off
  )
  
  # Update counter: increment if balanced, reset to 0 if not
  env.extras["balanced_stance_counter"] = torch.where(
    in_balanced_stance,
    env.extras["balanced_stance_counter"] + env.step_dt,  # Accumulate time
    torch.zeros_like(env.extras["balanced_stance_counter"]),  # Reset
  )
  
  # Reward scales with duration (clamped to prevent explosion)
  stance_duration = env.extras["balanced_stance_counter"]
  reward = torch.tanh(stance_duration / 3.0)  # Saturates at ~3 seconds
  
  return reward


def center_of_mass_stability(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward for keeping center of mass over the standing foot.
  
  Measures horizontal distance between CoM and standing foot, rewarding
  when CoM is well-positioned for stable balancing.
  
  Args:
    env: The environment instance.
    
  Returns:
    Tensor of shape (num_envs,) with reward values.
  """
  asset: Entity = env.scene["robot"]
  
  # Get standing leg choice from extras
  standing_leg = env.extras.get("standing_leg_choice", torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
  
  # Get body IDs for ankles (approximate foot position)
  left_ankle_ids, _ = asset.find_bodies(["left_ankle_roll_link"])
  right_ankle_ids, _ = asset.find_bodies(["right_ankle_roll_link"])
  
  # Get standing foot position
  left_ankle_pos = asset.data.body_link_pos_w[:, left_ankle_ids[0], :2]  # x, y only
  right_ankle_pos = asset.data.body_link_pos_w[:, right_ankle_ids[0], :2]
  
  standing_foot_pos = torch.where(
    standing_leg.unsqueeze(1) == 0,
    left_ankle_pos,
    right_ankle_pos,
  )
  
  # Get center of mass position (use root as approximation)
  com_pos = asset.data.root_link_pos_w[:, :2]  # x, y only
  
  # Calculate horizontal distance between CoM and standing foot
  horizontal_distance = torch.norm(com_pos - standing_foot_pos, dim=1)
  
  # Reward inversely proportional to distance (exponential falloff)
  # Ideal distance is ~0.1m (foot width), penalize as it gets further
  reward = torch.exp(-horizontal_distance / 0.15)
  
  return reward


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


def static_stance_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalty for base linear velocity (encourages static balance, not walking).
  
  Tree pose should be a STATIC balanced stance, not locomotion on one foot.
  This penalizes horizontal movement of the base.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs,) with penalty values (positive, to be negated by weight).
  """
  asset: Entity = env.scene[asset_cfg.name]
  lin_vel = asset.data.root_link_lin_vel_b
  
  # L2 norm of horizontal (x, y) velocity - penalize horizontal movement
  # Don't penalize vertical velocity (natural oscillation during balancing)
  horizontal_vel = lin_vel[:, :2]  # x, y components only
  return torch.sum(torch.square(horizontal_vel), dim=1)


def falling_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  sensor_name: str = "undesired_body_ground_contact",
) -> torch.Tensor:
  """Heavy penalty when robot is actually falling (non-foot body parts touching ground).
  
  This detects ACTUAL falling by checking a contact sensor that monitors
  non-foot body parts for ground contact. This prevents the robot from gaming
  the system by maintaining upright orientation while falling.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    sensor_name: Name of the contact sensor monitoring unwanted ground contacts.
    
  Returns:
    Tensor of shape (num_envs,) - penalty (1.0) if fallen, else 0.
  """
  asset: Entity = env.scene[asset_cfg.name]
  
  # Check the contact sensor for non-foot body ground contact
  # Sensor returns > 0 if any contact is detected
  contact_data = asset.data.sensor_data[sensor_name]  # Shape: (num_envs, num_contacts)
  
  # If ANY contact is detected, robot has fallen
  has_contact = contact_data[:, 0] > 0
  
  return has_contact.float()


def orientation_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Strong penalty for any tilting away from upright.
  
  This provides a continuous penalty that increases as the robot tilts,
  giving strong feedback to prevent falls before they happen.
  
  Args:
    env: The environment instance.
    asset_cfg: The asset configuration.
    
  Returns:
    Tensor of shape (num_envs,) with penalty values (positive, to be negated by weight).
  """
  asset: Entity = env.scene[asset_cfg.name]
  projected_gravity = asset.data.projected_gravity_b
  
  # Calculate tilt angle from upright
  # When upright, projected_gravity = [0, 0, -1], so z-component is -1
  # As robot tilts, z-component moves away from -1
  upright_error = torch.abs(projected_gravity[:, 2] + 1.0)
  
  # Exponential penalty that grows rapidly as robot tilts
  # At 0° tilt: penalty ≈ 0
  # At 20° tilt: penalty ≈ 0.5
  # At 40° tilt: penalty ≈ 2.0
  # This gives strong signal before falling
  penalty = torch.exp(upright_error * 5.0) - 1.0
  
  return penalty


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

