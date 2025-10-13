"""G1-specific configuration for single-leg balancing task."""

from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE, G1_ROBOT_CFG
from mjlab.tasks.balancing.balancing_env_cfg import BalancingEnvCfg
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class G1BalancingEnvCfg(BalancingEnvCfg):
  def __post_init__(self):
    # Configure contact sensors for feet only
    # These detect which foot is on the ground vs off the ground
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

    # Disable pushes initially - robot needs to learn lifting in stable conditions first
    self.events.push_robot = None
    
    # Disable foot friction randomization
    self.events.foot_friction = None

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

