"""G1-specific configuration for single-leg balancing task."""

from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.tasks.balancing.balancing_env_cfg import BalancingEnvCfg
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class G1BalancingEnvCfg(BalancingEnvCfg):
  def __post_init__(self):
    # Configure contact sensors for both feet
    foot_contact_sensors = [
      ContactSensorCfg(
        name=f"{side}_foot_ground_contact",
        body1=f"{side}_ankle_roll_link",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
      for side in ["left", "right"]
    ]
    g1_cfg = replace(G1_ROBOT_CFG, sensors=tuple(foot_contact_sensors))

    self.scene.entities = {"robot": g1_cfg}
    self.actions.joint_pos.scale = G1_ACTION_SCALE

    # Configure foot friction randomization
    self.events.foot_friction = None  # Disable for simplicity in balancing task

    # Configure joint posture reward with standard deviations
    self.rewards.joint_posture.params["std"] = {
      # Lower body - allow more flexibility for balancing
      r".*hip_pitch.*": 0.4,
      r".*hip_roll.*": 0.2,
      r".*hip_yaw.*": 0.2,
      r".*knee.*": 0.5,  # More flexibility for raised knee
      r".*ankle_pitch.*": 0.3,
      r".*ankle_roll.*": 0.15,
      # Waist - keep relatively stable
      r".*waist_yaw.*": 0.15,
      r".*waist_roll.*": 0.1,
      r".*waist_pitch.*": 0.1,
      # Arms - allow movement for balance
      r".*shoulder_pitch.*": 0.4,
      r".*shoulder_roll.*": 0.2,
      r".*shoulder_yaw.*": 0.15,
      r".*elbow.*": 0.3,
      r".*wrist.*": 0.4,
    }

    self.viewer.body_name = "torso_link"


@dataclass
class G1BalancingEnvCfg_PLAY(G1BalancingEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Disable observation corruption for evaluation
    self.observations.policy.enable_corruption = False

    # Disable push events for clean evaluation
    self.events.push_robot = None

    # Effectively infinite episode length for evaluation
    self.episode_length_s = int(1e9)

