# v3: STABILITY-FIRST - The Final Solution

## The Core Problem

**User feedback**: "The feet move to the maximum and the robot still falls. I want the robot to stand stable on one foot, not fall."

**Root cause**: We were rewarding knee height too aggressively (5.0 weight), causing the robot to optimize for maximum foot lift at the expense of stability.

## The Solution: Stability Must DOMINATE

The robot was learning: "Lift foot as high as possible" ✗
The robot needs to learn: "Stand stable on one foot, with foot moderately lifted" ✓

### New Reward Structure

```
CRITICAL - Stability (8.0 total = 76% of positive rewards):
├── upright: 4.0                      ← DOMINANT
├── center_of_mass_stability: 2.0     ← Critical for single-leg
└── balanced_stance_duration: 2.0     ← Sustaining stability

SECONDARY - Moderate Lifting (2.5 total = 24% of positive rewards):
├── knee_height: 1.5                  ← Reduced from 5.0!
└── foot_clearance: 1.0               ← Hint to lift

PENALTIES - Prevent Falling (-2.6 total):
├── orientation_penalty: -2.0 NEW     ← Exponential penalty for tilting
├── base_angular_penalty: -0.1        ← Excessive rotation
├── action_rate_l2: -0.01
└── joint_pos_limits: -0.5
```

**Stability DOMINATES by 3.2x** (8.0 vs 2.5)

### New Orientation Penalty (Critical Addition)

```python
def orientation_penalty(env):
    """Strong exponential penalty for tilting - prevents falls BEFORE they happen"""
    upright_error = abs(projected_gravity[:, 2] + 1.0)
    penalty = exp(upright_error * 5.0) - 1.0
    return penalty  # Weight: -2.0
```

**Effect**:
- 0° tilt: penalty ≈ 0
- 10° tilt: penalty ≈ 0.1 → -0.2 weighted
- 20° tilt: penalty ≈ 0.6 → -1.2 weighted  
- 30° tilt: penalty ≈ 1.4 → -2.8 weighted
- 40° tilt: penalty ≈ 2.7 → -5.4 weighted (HUGE!)

This gives strong feedback BEFORE the robot falls completely.

### Conservative Knee Height Stages

**Old (v2)**: Aggressive stages, max reward 3.3 at 0.40m
**New (v3)**: Conservative stages, max reward 2.5 at 0.40m

| Height | Old Reward | **New Reward** | Change |
|--------|-----------|----------------|--------|
| 0.10m  | 1.43      | **0.63**       | -56% (less aggressive) |
| 0.20m  | 2.17      | **0.86**       | -60% |
| 0.30m  | 2.95      | **1.62**       | -45% |
| 0.40m  | 3.29      | **2.48**       | -25% |

```python
# Stage 1: 0-0.20m (gentle lift) - max ~1.0
stage1 = 1.0 * (1.0 - exp(-height / 0.10))

# Stage 2: 0.20-0.35m (controlled) - max 1.0
stage2 = clamp((height - 0.20) / 0.15, 0, 1)

# Stage 3: 0.35m+ (modest bonus) - max 0.5
stage3 = clamp((height - 0.35) / 0.10, 0, 0.5)
```

The robot is encouraged to lift moderately, not aggressively.

### Stricter Terminations

**Old**: Single termination at 70° (very lenient)
**New**: Two-tier termination

```python
fell_over: 50°         # Stricter than before
excessive_tilt: 40°    # Early warning termination
```

Robot can't learn to "lean and lift" - episodes end before catastrophic falls.

## Why This Works

### Problem Breakdown

1. **Old v2**: knee_height=5.0, stability=3.0 → Robot optimizes height, falls
2. **New v3**: stability=8.0, knee_height=1.5 → Robot optimizes stability, lifts carefully

### Learning Progression

With stability dominating, the robot will learn:

1. **Phase 1 (Steps 0-2000)**: Stay perfectly upright
   - upright reward: 4.0 (immediately available)
   - Robot learns to maintain vertical orientation
   
2. **Phase 2 (Steps 2000-5000)**: Shift weight carefully
   - center_of_mass_stability: 2.0 kicks in
   - Robot learns to shift CoM over standing foot
   
3. **Phase 3 (Steps 5000-10000)**: Gentle foot lift
   - knee_height: 1.5 provides moderate incentive
   - foot_clearance: 1.0 hints to lift
   - Robot lifts foot WHILE maintaining stability
   
4. **Phase 4 (Steps 10000+)**: Sustain pose
   - balanced_stance_duration: 2.0 accumulates
   - Robot learns to hold stable single-leg stance

### Fail-Safe Mechanisms

1. **Orientation penalty (-2.0)**: Prevents tilting before it becomes a fall
2. **Early terminations (40-50°)**: Stops episodes before learning bad behaviors
3. **Dominant stability rewards (8.0)**: Always more valuable than lifting (2.5)

## Expected Training Curves

### Rewards
- `upright`: Should saturate at ~4.0 quickly and stay there
- `center_of_mass_stability`: Gradual improvement to ~2.0
- `balanced_stance_duration`: Slow growth as episodes last longer
- `knee_height`: Gradual increase to ~1.5-2.0 (NOT aggressive spikes)
- `orientation_penalty`: Should stay near 0 (low tilting)

### Terminations
- `time_out`: Should dominate (>80%)
- `fell_over` + `excessive_tilt`: Should decrease over time (<20% → <5%)
- Robot should NOT be constantly falling

### Behavior
- Robot should maintain **near-perfect vertical alignment**
- Foot should lift **slowly and carefully** (not aggressively)
- Robot should **hold the pose** for seconds at a time
- **No aggressive movements** that cause instability

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Key Insight

**The task is "single-leg BALANCING" not "maximum knee HEIGHT"**

By making stability rewards 3.2x larger than lifting rewards, we ensure the robot learns the RIGHT behavior: standing stable on one foot with moderate lift, rather than lifting aggressively and falling.

## Comparison Summary

| Aspect | v2 (Failed) | v3 (Stable) |
|--------|-------------|-------------|
| Knee height weight | 5.0 | 1.5 |
| Stability weights | 3.0 | 8.0 |
| Ratio | 1.7:1 lift | 3.2:1 stable |
| Max knee reward | 3.3 | 2.5 |
| Orientation penalty | None | -2.0 (exponential) |
| Termination angles | 70° | 40°/50° (two-tier) |
| Expected behavior | Aggressive lift → Fall | Stable stance → Gentle lift |

The robot will now optimize for what you actually want: **stable single-leg standing**.


