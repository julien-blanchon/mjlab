# Single-Leg Balancing Task

This task trains a Unitree G1 humanoid robot to balance on a single leg with the raised knee above a specified height threshold.

## Task Description

The robot must:
- Randomly select either the left or right leg as the standing leg at episode start
- Lift the opposite leg with the knee at least 0.4m above the ground
- Maintain balance for the entire episode duration (10 seconds)
- Avoid contact between the raised foot and the ground

## Gym Environments

Two environments are registered:

1. **`Mjlab-Balancing-Flat-Unitree-G1`**: Training environment with observation noise, robot pushes, and domain randomization
2. **`Mjlab-Balancing-Flat-Unitree-G1-Play`**: Evaluation environment with clean observations and no disturbances

## Training

Train the agent using:

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Evaluation

Evaluate a trained policy:

```bash
uv run play --task Mjlab-Balancing-Flat-Unitree-G1-Play --wandb-run-path your-org/mjlab/run-id
```

## Observations

The policy receives:
- Base angular velocity (3D, noisy)
- Projected gravity (3D, noisy)
- Joint positions relative to default (29D, noisy)
- Joint velocities (29D, noisy)
- Last action (29D)
- Standing leg indicator (2D one-hot: [1,0] for left, [0,1] for right)
- Raised knee height (1D, in meters)

Total observation dimension: 96

## Rewards

The reward function is designed to encourage stable single-leg balancing:

| Term | Weight | Description |
|------|--------|-------------|
| `alive` | 1.0 | Constant reward for staying balanced |
| `knee_height` | 2.0 | Exponential reward for keeping raised knee above 0.4m |
| `upright` | 0.5 | Reward for maintaining upright torso orientation |
| `base_stability` | -0.05 | Penalty for excessive angular velocity |
| `joint_posture` | 0.3 | Reward for staying close to default joint positions |
| `action_rate_l2` | -0.01 | Penalty for action changes (smoothness) |
| `joint_pos_limits` | -1.0 | Penalty for approaching joint limits |

## Terminations

Episodes terminate if:
- Time out after 10 seconds (default)
- Robot falls over (projected gravity angle > 70°)
- Raised foot makes contact with the ground
- Robot base drops below 0.3m height

## Implementation Details

### File Structure

```
balancing/
├── __init__.py
├── balancing_env_cfg.py          # Base configuration
├── mdp/
│   ├── __init__.py
│   ├── events.py                  # Standing leg randomization
│   ├── observations.py            # Standing leg indicator, knee height
│   ├── rewards.py                 # All reward terms
│   └── terminations.py            # Raised foot contact detection
└── config/
    └── g1/
        ├── __init__.py            # Gym registration
        ├── flat_env_cfg.py        # G1-specific config
        └── rl_cfg.py              # PPO configuration (reused from tracking task)
```

### Key Features

1. **Random leg selection**: At each episode reset, a random leg (left or right) is chosen as the standing leg
2. **Contact sensing**: Contact sensors on both feet detect if the raised foot touches the ground
3. **Knee height tracking**: Real-time monitoring of the raised knee's height above ground
4. **No external commands**: Unlike velocity or tracking tasks, this is a pure balancing task without command following

### Design Decisions

- **Episode length**: 10 seconds allows ~1-2 seconds for transition to single-leg stance and 5+ seconds of balancing
- **Push disturbances**: Applied every 2-4 seconds with velocity perturbations to test robustness
- **Simple scene**: Flat terrain only (no rough terrain or obstacles)
- **Flexible joint postures**: Higher standard deviations for balancing-critical joints (knees, hips, shoulders)
- **No curriculum**: Task difficulty is fixed to simplify training

## Notes

- This task reuses the `G1FlatPPORunnerCfg` from the tracking task for consistency
- The standing leg choice is stored in `env.extras["standing_leg_choice"]` (0 = left, 1 = right)
- Observation noise is applied during training to improve robustness
- The PLAY variant disables noise and disturbances for clean evaluation

