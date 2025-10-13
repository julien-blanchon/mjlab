# Single-Leg Balancing Task: Balance-Focused Updates

## Summary

Based on training observations showing the robot was learning foot movements but failing to maintain equilibrium, we've **completely redesigned the reward structure to prioritize balance** as the core skill.

## Key Changes

### 1. 🕐 Extended Episode Duration (10s → 20s)

**Rationale**: The robot needs time to:
- Transition to single-leg stance (~2-3 seconds)
- **Learn and practice maintaining balance** (~15+ seconds)
- Receive sufficient reward signal for sustained balancing

**Previous**: 10 seconds (too short - robot failed just after lifting)
**Current**: 20 seconds (allows proper balance learning)

### 2. 🎯 Balance-First Reward Hierarchy

Completely restructured rewards to emphasize **equilibrium over foot height**:

#### Core Balance Rewards (60% of positive reward - 5.0 total weight)

| Reward | Weight | Purpose |
|--------|--------|---------|
| `upright` | 2.5 | **Highest priority** - keeps torso vertical |
| `balanced_stance_duration` | 1.5 | **NEW** - Rewards sustaining pose over time (resets if foot drops) |
| `center_of_mass_stability` | 1.0 | **NEW** - Rewards CoM positioning over standing foot |

#### Foot Lifting Rewards (40% of positive reward - 4.5 total weight)

| Reward | Weight | Purpose |
|--------|--------|---------|
| `knee_height` | 3.0 | Continuous reward 0→0.4m (reduced from 4.0) |
| `foot_clearance` | 1.5 | Binary reward for any foot lift (reduced from 2.0) |

#### Auxiliary Rewards (Minimal interference)

All penalties drastically reduced to avoid fighting exploration:
- `base_stability`: -0.01 (was -0.02)
- `joint_posture`: 0.02 (was 0.05)
- `action_rate_l2`: -0.002 (was -0.005)
- `joint_pos_limits`: -0.3 (was -0.5)

### 3. 💪 Enhanced Push Disturbances

**Purpose**: Force the robot to learn robust balancing, not just static poses

**Previous**:
- Interval: 2.0-4.0 seconds
- Linear velocity: ±0.3 m/s
- No rotational disturbances

**Current**:
- Interval: 1.5-3.0 seconds (more frequent)
- Linear velocity: ±0.5 m/s (67% stronger)
- Angular velocity: ±0.3 rad/s (**NEW** - tests rotational stability)

### 4. 🆕 New Reward Functions

#### `balanced_stance_duration`
```python
# Tracks time spent with:
# - Raised foot off ground
# - Standing foot on ground
# - Counter resets to 0 if foot touches down

stance_duration = env.extras["balanced_stance_counter"]
reward = torch.tanh(stance_duration / 2.0)  # Saturates at ~2s
```

**Why this matters**: Teaches the robot that **sustaining** the balance is more important than just achieving it momentarily.

#### `center_of_mass_stability`
```python
# Measures horizontal distance between CoM and standing foot
horizontal_distance = ||com_pos - standing_foot_pos||
reward = exp(-horizontal_distance / 0.15)
```

**Why this matters**: This is a **fundamental balance principle** - the CoM must be over the base of support. This gives the robot direct feedback on proper weight distribution.

## Expected Training Improvements

### Before (Old Reward Structure)
- ❌ Robot moved feet correctly but fell immediately
- ❌ `knee_height` reward was sparse (rare spikes)
- ❌ No reward for actually maintaining balance
- ❌ Episodes too short to learn equilibrium

### After (Balance-First Structure)
- ✅ **Balance rewards dominate** (60% of positive weight)
- ✅ **Duration-based reward** encourages sustaining stance
- ✅ **CoM stability** teaches proper weight distribution
- ✅ **Longer episodes** allow practice maintaining equilibrium
- ✅ **Stronger disturbances** force robust balance learning
- ✅ Foot lifting is secondary (happens naturally once balanced)

## Reward Curves You Should See

1. **`upright`** (2.5): Should be high from the start, stay high
2. **`balanced_stance_duration`** (1.5): Should grow gradually as robot learns to sustain poses
3. **`center_of_mass_stability`** (1.0): Should improve as robot learns weight shifting
4. **`foot_clearance`** (1.5): Quick initial spike when foot lifts
5. **`knee_height`** (3.0): Gradual growth, less emphasis than before

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Key Insights

1. **Balance is a skill, not a state**: The old reward structure treated balance as automatic if the robot hit certain poses. The new structure recognizes balance as a continuous control task.

2. **Time matters**: By rewarding duration of balanced stance, we teach the robot that stability over time is the objective, not just achieving the configuration.

3. **Physics-based rewards work**: `center_of_mass_stability` directly encodes the physics of balance, giving the robot clear guidance on what "good balance" means.

4. **Hierarchy prevents local optima**: By making balance rewards dominant (5.0) over foot height (4.5), the robot is forced to prioritize stability, preventing solutions where it lifts the foot but immediately falls.

## Summary of All Files Changed

- `balancing_env_cfg.py`: Updated episode length, reward structure, push disturbances
- `mdp/rewards.py`: Added `balanced_stance_duration` and `center_of_mass_stability`
- `mdp/events.py`: Added counter reset on episode start
- `README.md`: Updated documentation with new reward philosophy

---

**The core philosophy**: Teach balance first, foot height second. A robot that can balance on one leg will naturally learn to lift the other foot higher once it has mastered the equilibrium skills.

