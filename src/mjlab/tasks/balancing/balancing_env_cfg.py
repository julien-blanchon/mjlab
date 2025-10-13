"""Single-leg balancing task configuration.

This module defines the base configuration for single-leg balancing tasks.
Robot-specific configurations are located in the config/ directory.
"""

import math
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import EventTermCfg as EventTerm
from mjlab.managers.manager_term_config import ObservationGroupCfg as ObsGroup
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import RewardTermCfg as RewardTerm
from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.balancing import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

##
# Scene.
##

SCENE_CFG = SceneCfg(
  terrain=TerrainImporterCfg(terrain_type="plane"),
  num_envs=1,
  extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  asset_name="robot",
  body_name="",  # Override in robot cfg.
  distance=3.0,
  elevation=-5.0,
  azimuth=90.0,
)

##
# MDP.
##


@dataclass
class ActionCfg:
  joint_pos: mdp.JointPositionActionCfg = term(
    mdp.JointPositionActionCfg,
    asset_name="robot",
    actuator_names=[".*"],
    scale=0.5,
    use_default_offset=True,
  )


@dataclass
class ObservationCfg:
  @dataclass
  class PolicyCfg(ObsGroup):
    base_ang_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity: ObsTerm = term(
      ObsTerm,
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel: ObsTerm = term(
      ObsTerm,
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    actions: ObsTerm = term(ObsTerm, func=mdp.last_action)

    def __post_init__(self):
      self.enable_corruption = True

  @dataclass
  class PrivilegedCfg(PolicyCfg):
    def __post_init__(self):
      super().__post_init__()
      self.enable_corruption = False

  policy: PolicyCfg = field(default_factory=PolicyCfg)
  critic: PrivilegedCfg = field(default_factory=PrivilegedCfg)


@dataclass
class EventCfg:
  reset_base: EventTerm = term(
    EventTerm,
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"yaw": (-0.2, 0.2)},
      "velocity_range": {},
    },
  )
  reset_robot_joints: EventTerm = term(
    EventTerm,
    func=mdp.reset_joints_by_scale,
    mode="reset",
    params={
      "position_range": (1.0, 1.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
    },
  )
  randomize_standing_leg: EventTerm = term(
    EventTerm,
    func=mdp.randomize_standing_leg,
    mode="reset",
    params={},
  )
  # Gentle pushes to encourage robust balance recovery
  push_robot: EventTerm = term(
    EventTerm,
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(8.0, 12.0),  # Infrequent - every 8-12 seconds
    params={
      "velocity_range": {
        "x": (-0.2, 0.2),      # Gentle linear pushes
        "y": (-0.2, 0.2),
        "yaw": (-0.1, 0.1),    # Gentle rotational disturbance
      }
    },
  )
  # Foot friction randomization disabled for stable learning
  foot_friction: EventTerm | None = None


@dataclass
class RewardCfg:
  # 1. Stay UPRIGHT - Most important for classical yoga pose
  upright: RewardTerm = term(
    RewardTerm,
    func=mdp.upright_posture,
    weight=3.0,  # HIGHEST - must maintain vertical orientation
  )
  
  # 2. Raise FOOT to MODERATE height (tree pose range)
  # NOTE: Measures foot/ankle height, not knee (prevents bending knee exploit)
  foot_height: RewardTerm = term(
    RewardTerm,
    func=mdp.raised_foot_height_optimal_range,
    weight=3.5,  # Increased to push for higher lift
    params={
      "min_height": 0.10,      # Start earlier (4 inches)
      "optimal_height": 0.25,  # Sweet spot (10 inches - classical tree pose)
      "max_height": 0.40,      # Higher limit before penalty (16 inches)
    },
  )
  
  # 3. Get one foot off ground (standing foot must stay on ground)
  foot_clearance: RewardTerm = term(
    RewardTerm,
    func=mdp.foot_clearance,
    weight=1.5,  # Initial signal to lift
  )
  
  # PENALTIES: Encourage static balance
  static_stance: RewardTerm = term(
    RewardTerm,
    func=mdp.static_stance_penalty,
    weight=-0.5,  # Penalize horizontal movement (walking on one foot)
  )
  action_rate_l2: RewardTerm = term(
    RewardTerm, 
    func=mdp.action_rate_l2, 
    weight=-0.01
  )


@dataclass
class TerminationCfg:
  # Time out after episode length
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  
  # Terminate if robot tilts too much (lenient to allow exploration)
  fell_over: DoneTerm = term(
    DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
  )


##
# Environment.
##

SIM_CFG = SimulationCfg(
  nconmax=140_000,
  njmax=300,
  mujoco=MujocoCfg(
    timestep=0.005,
    iterations=10,
    ls_iterations=20,
  ),
)


@dataclass
class BalancingEnvCfg(ManagerBasedRlEnvCfg):
  scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
  observations: ObservationCfg = field(default_factory=ObservationCfg)
  actions: ActionCfg = field(default_factory=ActionCfg)
  rewards: RewardCfg = field(default_factory=RewardCfg)
  events: EventCfg = field(default_factory=EventCfg)
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  commands: None = None  # No commands for balancing task
  curriculum: None = None  # No curriculum for balancing task
  sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
  viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
  decimation: int = 4  # 50 Hz control frequency.
  episode_length_s: float = 20.0  # Sufficient time to learn and maintain single-leg stance

