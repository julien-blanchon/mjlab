# v4: FORCE FOOT LIFTING - The Solution

## The Problem (from training data)

Looking at your curves, the robot was **staying on both feet**:
- `balanced_stance_duration`: ~0.03-0.05 (should be >0.5 if sustained)
- `foot_clearance`: ~0.8-0.9 (should be 1.0 consistently)
- `upright`: ~3.5-4.0 (saturated)
- `center_of_mass_stability`: ~1.2-1.4

**Root cause**: The robot found a local optimum where it could get HIGH rewards (6.5+) by just standing on both feet with good posture, without ever needing to lift a foot!

## The Core Issue

```
OLD v3 Rewards (standing on both feet):
  upright: 4.0 ✓ (always available)
  center_of_mass_stability: 2.0 ✓ (easy with both feet)
  alive: 0.5 ✓
  Total: 6.5 reward with ZERO risk!

Standing on one foot:
  upright: 4.0 
  center_of_mass_stability: 2.0
  knee_height: 1.5
  foot_clearance: 1.0
  balanced_stance_duration: 2.0
  Total: 10.5 reward but WITH risk of falling

→ Robot chose the safe 6.5 over risky 10.5
```

## The Solution: Conditional Rewards

### Key Innovation: `conditional_upright_posture`

```python
def conditional_upright_posture(env):
    """Reward for upright posture ONLY when foot is OFF ground.
    
    Prevents robot from getting upright reward while on both feet.
    FORCES robot to lift foot to access this reward.
    """
    # Check if raised foot is off ground
    raised_foot_off_ground = check_foot_contact()
    
    # Calculate upright reward
    upright_reward = exp(-upright_error / 0.25)
    
    # Only give reward if foot is lifted!
    return upright_reward if raised_foot_off_ground else 0.0
```

### New Reward Structure

```
PRIMARY - Lifting Required (9.0 total):
├── foot_clearance: 3.0           ← Immediate reward for lifting
├── knee_height: 3.0               ← Progressive height rewards
└── balanced_stance_duration: 3.0  ← Sustained single-leg stance

SECONDARY - Conditional (3.0 total):
├── conditional_upright: 2.0       ← ONLY if foot off ground!
└── center_of_mass_stability: 1.0

PENALTIES (light):
├── orientation_penalty: -1.0
└── others: -0.35
```

### Reward Comparison

| Scenario | Old v3 | **New v4** | Change |
|----------|--------|------------|--------|
| **Both feet on ground** | 6.5 | **1.3** | -80% ✓ |
| **One foot lifted** | 10.5 | **12.0** | +14% ✓ |
| **Ratio** | 1.6x | **9.2x** | Huge! |

Standing on both feet is now only worth **1.3 reward**.
Standing on one foot is now worth **12.0 reward**.

**The robot MUST lift the foot to get meaningful rewards!**

## Additional Changes

### 1. Increased Lifting Rewards
- `foot_clearance`: 1.0 → 3.0 (3x increase)
- `knee_height`: 1.5 → 3.0 (2x increase)
- `balanced_stance_duration`: 2.0 → 3.0 (1.5x increase)

### 2. Relaxed Terminations
- `fell_over`: 50° → 60° (more lenient)
- `excessive_tilt`: Removed (was terminating too early)

Allows more exploration without premature termination.

### 3. Reduced Penalties
- `orientation_penalty`: -2.0 → -1.0 (less harsh)
- `base_angular_penalty`: -0.1 → -0.05 (allow movement)

Robot can explore without excessive penalties.

## Why This Works

### The Math
```
Standing on both feet:
  conditional_upright: 0.0 (foot not lifted - NO REWARD!)
  center_of_mass: ~1.0
  alive: 0.3
  Total: ~1.3 reward

Lifting foot:
  conditional_upright: 2.0 (foot off ground - UNLOCKED!)
  foot_clearance: 3.0
  knee_height: ~2.0-3.0 (depending on height)
  balanced_stance_duration: ~0-3.0 (grows with time)
  center_of_mass: ~1.0
  alive: 0.3
  Total: ~8.3-11.3 reward (6.4x - 8.7x more!)
```

### The Psychology
1. **Early training**: Robot tries random actions
2. **Discovery**: Accidentally lifts foot → gets 8+ reward (vs 1.3)
3. **Learning**: "Lifting foot = much higher reward!"
4. **Optimization**: Learns to sustain lifted pose for max reward

## Expected Training Behavior

### Rewards
- `foot_clearance`: Should spike to ~3.0 quickly (binary)
- `knee_height`: Gradual growth to 2.0-3.0
- `conditional_upright`: Should grow from 0 → 2.0 (as foot lifts)
- `balanced_stance_duration`: Gradual increase (0 → 1.0+)
- `center_of_mass_stability`: Stable ~1.0

### Terminations
- `fell_over`: Should be moderate initially, then decrease
- `time_out`: Should dominate as learning progresses

### Key Indicator
**Watch `conditional_upright`**: 
- If it stays at 0: Robot not lifting (shouldn't happen with v4)
- If it grows to ~2.0: Robot is lifting! ✓

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Summary

| Version | Issue | Solution |
|---------|-------|----------|
| v1-v2 | Episodes too short | Removed root_too_low |
| v3 | Robot stays on both feet | **v4: Conditional rewards** |

**v4 Fix**: Robot CANNOT get high rewards without lifting foot. The `conditional_upright` reward (2.0) is ONLY available when foot is off ground, making single-leg stance 9.2x more rewarding than both feet!

This forces the robot to learn single-leg balancing as the optimal policy.


