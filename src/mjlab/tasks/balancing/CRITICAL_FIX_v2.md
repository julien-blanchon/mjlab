# Critical Fix v2: Addressing Episode Termination Issues

## Problem Diagnosis

Training graphs revealed the **root cause** of learning failure:

### Termination Analysis
- **`root_too_low`: 5-6 per episode** ← DOMINANT FAILURE MODE
- `fell_over`: ~0.1 per episode  
- `raised_foot_contact`: Nearly 0
- `time_out`: Nearly 0

### Reward Analysis
- `knee_height`: Still sparse spikes (robot barely lifts)
- `balanced_stance_duration`: ~0.001 (essentially 0)
- Robot couldn't learn because episodes ended too quickly

## Root Cause

**The robot naturally crouches when lifting a leg** (to redistribute weight and lower center of mass). The 0.3m `root_too_low` threshold was **terminating episodes before the robot could learn anything**!

Think of it like learning to stand on one leg as a human:
1. You shift weight to one side
2. You naturally bend your knees slightly
3. **You crouch down a bit to stabilize**
4. THEN you lift the other foot

We were killing the episode at step 3!

## Critical Fixes Applied

### 1. ❌ REMOVED `root_too_low` Termination

**Before**: Terminated at 0.3m height (5-6 times per episode)
**After**: REMOVED - allows robot to crouch naturally during lifting

### 2. ❌ REMOVED `raised_foot_contact` Termination  

**Reason**: Too strict during exploration phase. Robot needs to experiment with different lifting strategies, including briefly touching down.

### 3. ✅ KEPT Safety Terminations

- `fell_over`: Still terminates if robot falls (angle > 70°)
- `time_out`: Episode boundary at 20 seconds

### 4. 🎯 REDESIGNED Knee Height Reward (STAGED)

**The Problem**: Previous "continuous" reward wasn't dense enough at low heights.

**New Solution**: Three-stage reward that provides STRONG signal at ALL heights:

```python
# Stage 1: 0-0.15m (Lift foot) - Strong exponential 
stage1_reward = 2.0 * (1.0 - exp(-height / 0.08))  # Saturates at ~2.0

# Stage 2: 0.15-0.30m (Raise knee) - Linear
stage2_reward = clamp((height - 0.15) / 0.15, 0, 1)  # +1.0 max

# Stage 3: 0.30m+ (Reach target) - Exponential bonus
stage3_reward = clamp(height - 0.30, 0, 0.3) * 3.0  # +0.9 max at 0.4m
```

**Reward Density**:
| Height | Old Reward | New Reward | Improvement |
|--------|-----------|-----------|-------------|
| 0.05m  | ~0.0      | 0.93      | ∞ better! |
| 0.10m  | ~0.0      | 1.43      | ∞ better! |
| 0.15m  | ~0.0      | 1.69      | ∞ better! |
| 0.20m  | ~0.25     | 2.17      | 8.7x |
| 0.30m  | ~0.75     | 2.95      | 3.9x |
| 0.40m  | 1.0       | 3.29      | 3.3x |

### 5. 🔄 REBALANCED Reward Hierarchy

**New Philosophy**: Foot lifting FIRST, balance SECOND

```
PRIMARY (Foot Lifting - 7.0 weight):
├── knee_height: 5.0 (staged, dense from 0m)
└── foot_clearance: 2.0 (binary signal)

SECONDARY (Balance - 3.0 weight):
├── upright: 1.5
├── balanced_stance_duration: 1.0
└── center_of_mass_stability: 0.5

AUXILIARY (Minimal):
├── alive: 0.2
├── joint_posture: 0.0 (DISABLED - was fighting lifting)
└── penalties: ≤ 0.005 each
```

**Rationale**: Balance is useless if the foot never lifts. Once the robot learns to lift, balance rewards naturally kick in.

## Expected Training Improvements

### Before v2
- ❌ Episodes terminated at 5-6 per episode (root_too_low)
- ❌ No learning time (died while crouching)
- ❌ Sparse knee_height reward (rare spikes)
- ❌ balanced_stance_duration ≈ 0 (no time to balance)

### After v2
- ✅ Episodes run full 20 seconds or until fell_over
- ✅ Robot can explore crouching strategies
- ✅ DENSE knee_height reward from first millimeter
- ✅ Time to practice balance after lifting
- ✅ 70% of reward weight on actual foot lifting

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Expected Reward Curves

You should now see:

1. **`knee_height`**: Smooth, continuous growth from step 1 (not sparse spikes!)
2. **Episode terminations**: Mostly time_out, occasional fell_over, NO root_too_low
3. **`foot_clearance`**: Quick saturation at ~2.0
4. **`upright`**: Stable around 1.5
5. **`balanced_stance_duration`**: Gradual increase as episodes last longer

## Why This Will Work

1. **No premature termination**: Robot has full 20 seconds to learn
2. **Dense reward signal**: Gets reward for EVERY bit of lifting (0.93 at 5cm!)
3. **Natural progression**: 
   - First learns to crouch (not punished)
   - Then learns to lift foot (strong reward 0-15cm)
   - Then learns to raise knee (reward 15-30cm)  
   - Then learns to reach target (bonus 30cm+)
   - Balance emerges naturally during this process
4. **Exploration-friendly**: Robot can experiment without instant death

## Summary

**v1 Problem**: "Balance first" approach failed because robot died before learning to lift
**v2 Solution**: "Lift first" approach with dense rewards and no premature termination

The task is now properly decomposed into learnable stages with dense feedback throughout.


