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
    standing_leg: ObsTerm = term(ObsTerm, func=mdp.standing_leg_indicator)
    raised_knee_height: ObsTerm = term(ObsTerm, func=mdp.raised_knee_height)

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
  push_robot: EventTerm | None = term(
    EventTerm,
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(2.0, 4.0),
    params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
  )


@dataclass
class RewardCfg:
  alive: RewardTerm = term(
    RewardTerm,
    func=mdp.alive_bonus,
    weight=1.0,
  )
  knee_height: RewardTerm = term(
    RewardTerm,
    func=mdp.knee_height_above_threshold,
    weight=2.0,
    params={"threshold": 0.4, "std": 0.1},
  )
  upright: RewardTerm = term(
    RewardTerm,
    func=mdp.upright_posture,
    weight=0.5,
  )
  base_stability: RewardTerm = term(
    RewardTerm,
    func=mdp.base_stability,
    weight=-0.05,
  )
  joint_posture: RewardTerm = term(
    RewardTerm,
    func=mdp.joint_posture,
    weight=0.3,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
      "std": [],
    },
  )
  action_rate_l2: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.01)
  joint_pos_limits: RewardTerm = term(RewardTerm, func=mdp.joint_pos_limits, weight=-1.0)


@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  fell_over: DoneTerm = term(
    DoneTerm, func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
  )
  raised_foot_contact: DoneTerm = term(
    DoneTerm,
    func=mdp.raised_foot_contact,
    params={},
  )
  root_too_low: DoneTerm = term(
    DoneTerm,
    func=mdp.root_height_below_minimum,
    params={"minimum_height": 0.3},
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
  episode_length_s: float = 10.0  # Allows transition + 5s balance

