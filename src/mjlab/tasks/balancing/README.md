# Single-Leg Balancing Task

This task trains a Unitree G1 humanoid robot to balance on a single leg with the raised knee above a specified height threshold.

## Task Description

The robot must learn a **classical tree pose** (yoga-style single-leg balance):
- Randomly select either the left or right leg as the standing leg at episode start
- **Maintain upright torso orientation** (primary objective - prevents backbend/extreme tilt)
- Lift the opposite leg with knee in optimal range (0.30-0.40m above ground)
- Sustain the balanced stance for extended periods (20 second episodes)
- Handle gentle push disturbances every 8-12s (±0.2 m/s linear, ±0.1 rad/s angular)
- Avoid excessive knee height (>0.45m is penalized as unstable)

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

The policy receives **proprioceptive observations only** (no explicit leg indicator):
- Base angular velocity (3D, noisy)
- Projected gravity (3D, noisy)
- Joint positions relative to default (29D, noisy)
- Joint velocities (29D, noisy)
- Last action (29D)

Total observation dimension: **93**

**Note:** The policy does NOT receive explicit information about which leg to lift. It must discover this from the reward signal and proprioceptive feedback. A random leg is selected at each episode reset via `randomize_standing_leg` event.

## Rewards

The reward function encourages a **classical static tree pose**:

### Primary Rewards (7.0 total weight)

| Term | Weight | Description |
|------|--------|-------------|
| `upright` | 3.0 | **DOMINANT** - Maintain vertical torso orientation (prevents backbend/tilt) |
| `foot_height` | 2.5 | Raised **foot** at 0.25-0.30m (optimal), penalizes >0.35m. **ONLY active in valid single-leg stance** |
| `foot_clearance` | 1.5 | Progressive 0.5→1.0 based on foot lift. **ONLY active in valid single-leg stance** |

### Penalties (-0.51 total)

| Term | Weight | Description |
|------|--------|-------------|
| `static_stance` | -0.5 | Penalizes horizontal base movement (encourages static balance, not walking) |
| `action_rate_l2` | -0.01 | Smoothness penalty |

### Anti-Exploit Mechanisms

**Critical:** Both `foot_height` and `foot_clearance` are **conditional rewards** that ONLY activate when:
- ✓ Raised foot is OFF ground  
- ✓ Standing foot is ON ground

This prevents THREE exploits:
1. **Jumping hack**: Lifting both feet → No reward (not valid single-leg stance)
2. **Sitting hack**: Both feet on ground → No reward (can't get foot_height rewards)
3. **Bending knee hack**: Measures foot height (not knee), so bending knee doesn't give extra reward
4. **Walking hack**: `static_stance` penalty (-0.5) discourages horizontal movement

### Reward Behavior

```python
Both feet on ground (sitting, stationary):
  upright: ~1.0 × 3.0 = 3.0 ✓
  foot_height: 0.0 ✗ (not valid stance)
  foot_clearance: 0.0 ✗ (not valid stance)
  static_stance: 0.0 × -0.5 = 0.0 ✓ (not moving)
  Total: ~3.0

Classical tree pose (upright, foot at 0.25m, stationary):
  upright: ~1.0 × 3.0 = 3.0 ✓✓✓
  foot_height: 1.0 × 2.5 = 2.5 ✓✓
  foot_clearance: ~0.9 × 1.5 = 1.35 ✓
  static_stance: ~0.0 × -0.5 = 0.0 ✓ (minimal movement)
  Total: ~6.85 ← OPTIMAL POSE

Walking on one foot (foot at 0.25m, moving):
  upright: ~0.9 × 3.0 = 2.7 ✓
  foot_height: 1.0 × 2.5 = 2.5 ✓
  foot_clearance: ~0.9 × 1.5 = 1.35 ✓
  static_stance: ~1.0 × -0.5 = -0.5 ✗ (PENALTY for movement!)
  Total: ~6.05 ← Worse than static

Extreme backbend (tilted, foot at 0.40m):
  upright: ~0.3 × 3.0 = 0.9 ✗ (lose 2.1 from tilt!)
  foot_height: 0.3 × 2.5 = 0.75 ✗ (decaying, too high!)
  foot_clearance: ~1.0 × 1.5 = 1.5 ✓
  static_stance: ~0.0 × -0.5 = 0.0
  Total: ~3.15 ← STRONGLY DISCOURAGED
```

**Clear gradient:** Walking (~6.05) < Backbend (~3.15) < Sitting (~3.0) < **Static tree pose (~6.85)**

## Terminations

Episodes terminate if:
- **Time out** after 20 seconds (sufficient time for stance + push recovery)
- **Robot falls over**: projected gravity angle > 70° (lenient to allow exploration)

**Note:** Previous versions had more terminations (`body_contact_with_ground`, `both_feet_off_ground`), but these caused issues:
- `body_contact_with_ground`: Triggered incorrectly, causing 1-step episodes
- `both_feet_off_ground`: Contact sensors had timing issues at reset

Anti-exploit mechanisms are now **built into the rewards** (conditional on valid single-leg stance) rather than terminations.

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

- **Episode length**: 20 seconds - time for stance establishment + push recovery practice
- **Push disturbances**: ENABLED - Gentle pushes every 8-12s (±0.2 m/s) for active balance training
- **Upright-dominant rewards**: `upright` (3.0) > `knee_height` (2.5) → forces vertical pose
- **Optimal height range**: Rewards 0.30-0.40m (classical tree pose), penalizes >0.45m (extreme poses)
- **Anti-exploit mechanisms**: 
  - All height-related rewards conditional on valid single-leg stance
  - Prevents: jumping (both feet off), sitting (both feet on), backbend (tilted + extreme height)
- **Simple scene**: Flat terrain only (no rough terrain or obstacles)
- **Light penalties**: 700:1 positive/negative ratio encourages exploration
- **No curriculum**: Fixed difficulty - robot learns tree pose from scratch

## Notes

- This task reuses the `G1FlatPPORunnerCfg` from the tracking task for consistency
- The standing leg choice is stored in `env.extras["standing_leg_choice"]` (0 = left, 1 = right)
- Observation noise is applied during training to improve robustness
- The PLAY variant disables noise and disturbances for clean evaluation

