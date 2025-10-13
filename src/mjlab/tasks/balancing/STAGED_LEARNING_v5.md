# v5: STAGED LEARNING WITH WEIGHT SHIFT - Complete Solution

## The Breakthrough Insight

**The robot can't lift a foot that has weight on it!**

Previous versions tried to reward lifting directly, but missed the critical prerequisite: **weight shift**. In human single-leg balancing, you:
1. **First** shift your weight onto one foot (unloading the other)
2. **Then** lift the unloaded foot (now easy!)
3. **Finally** maintain balance

v5 implements this natural progression.

## Core Changes

### 1. New Stage 1 Reward: Weight Shift Preparation (4.0)

```python
def weight_shift_preparation(env):
    """Reward for shifting CoM over standing foot BEFORE lifting.
    
    Key insight: Robot must unload the foot before it can lift it!
    """
    # Calculate horizontal distance from CoM to standing foot
    distance = ||com_xy - standing_foot_xy||
    
    # Sharp reward for tight alignment (within 5cm = excellent)
    return exp(-distance / 0.05)
```

This teaches the robot to shift weight FIRST, making lifting natural.

### 2. Pushes Disabled Initially

**Before (v4)**: Pushes every 1.5-3.0s with ±0.5 m/s velocity
**Now (v5)**: Pushes DISABLED (`self.events.push_robot = None`)

Why? The robot needs a **stable environment** to discover that:
- Weight shift → 4.0 reward
- Weight shift + foot lift → 12+ reward

Random pushes during early training prevent this discovery.

### 3. Reduced Knee Height Threshold

**Before**: 0.4m (16 inches - quite high!)
**Now**: 0.25m (10 inches - more discoverable)

Start easier, then increase via curriculum once lifting is learned.

### 4. Very High Foot Clearance Reward

**Before**: 1.0 → 3.0
**Now**: **5.0** (highest)

This is the critical breakthrough moment - when foot first leaves ground, the robot needs a HUGE signal!

## Complete Reward Structure

```
STAGE 1 - Weight Shift (4.0 total):
├── weight_shift_preparation: 4.0    ← NEW! Teaches prerequisite

STAGE 2 - Initial Lift (5.0 total):
├── foot_clearance: 5.0               ← Breakthrough signal

STAGE 3 - Height + Duration (7.0 total):
├── knee_height: 3.0                  ← Progressive height (threshold=0.25m)
└── balanced_stance_duration: 4.0     ← Sustaining the pose

CONDITIONAL - Only when foot up (2.0 total):
└── conditional_upright: 2.0          ← Prevents two-foot strategy

AUXILIARY (0.2 total):
└── alive: 0.2

PENALTIES (-0.72 total, very light):
├── orientation_penalty: -0.5
├── base_angular_penalty: -0.02
├── action_rate: -0.002
└── joint_limits: -0.2
```

**Total positive rewards**: 18.0
**Total penalties**: -0.72
**Net focus**: Learning, not penalizing

## Reward Progression

| Action | Reward | Cumulative |
|--------|--------|------------|
| Standing normally | 0.2 (alive) | 0.2 |
| **Shift weight over standing foot** | +4.0 | **4.2** ✓ |
| **Lift foot off ground** | +5.0 | **9.2** ✓✓ |
| **Lift knee to 0.15m** | +1.5 | **10.7** ✓✓✓ |
| **Lift knee to 0.25m** | +3.0 | **12.2** ✓✓✓ |
| **Unlock conditional_upright** | +2.0 | **14.2** ✓✓✓✓ |
| **Sustain for 1 second** | +0.5 | **14.7** ✓✓✓✓ |
| **Sustain for 5 seconds** | +2.5 | **17.2** ✓✓✓✓✓ |

The reward **guides the robot step-by-step** through the skill.

## Why v5 Will Work

### Problem with v1-v4
```
Random exploration → Foot lift attempts (rare!)
                   ↓
                Robot falls (weight still on foot!)
                   ↓
                Negative experience → avoid lifting
```

### v5 Solution
```
Random exploration → Weight shift (common!)
                   ↓
                +4.0 reward → learn weight shift
                   ↓
                Foot unloaded → accidental lift (easier!)
                   ↓
                +5.0 reward → huge signal!
                   ↓
                Learn: weight shift + lift = 12+ reward
```

### The Math
- **Both feet on ground, no weight shift**: 0.2 reward
- **Both feet on ground, good weight shift**: 4.2 reward (21x more!)
- **One foot lifted**: 12-17 reward (3x more than weight shift!)

The gradient is **huge and continuous**!

## Expected Training Behavior

### Early Training (0-100k steps)
- `weight_shift_preparation`: 0 → 3.0-4.0 (robot learns CoM control)
- `foot_clearance`: 0 (still on both feet)
- `knee_height`: 0
- Robot should be swaying, shifting weight

### Breakthrough (100k-500k steps)
- `foot_clearance`: Sudden spikes to 5.0! (discovery!)
- `knee_height`: Starts growing (0.5 → 1.0 → 2.0)
- `conditional_upright`: 0 → 2.0 (foot is up!)
- `balanced_stance_duration`: Starts accumulating

### Mastery (500k+ steps)
- `weight_shift_preparation`: Stable ~4.0
- `foot_clearance`: Stable ~5.0
- `knee_height`: Reaches ~3.0 (knee at 0.25m)
- `balanced_stance_duration`: Growing steadily
- `time_out` termination dominates (good sign!)

## Key Monitoring Metrics

**Critical Success Indicators**:
1. `weight_shift_preparation`: Should reach 3.5-4.0 in first 50k steps
2. `foot_clearance`: Should spike to 5.0 between 100k-500k steps (the "aha!" moment)
3. `conditional_upright`: Should follow foot_clearance (proves foot is lifting)
4. `balanced_stance_duration`: Should grow linearly after foot_clearance activates

**Red Flags**:
- `weight_shift_preparation` stuck at <2.0: Robot not learning CoM control
- `foot_clearance` stays at 0 after 1M steps: Reward might still not be discoverable
- `fell_over` termination dominates: Need even gentler penalties

## Differences from v4

| Feature | v4 | v5 |
|---------|----|----|
| **Key Innovation** | Conditional upright | **Weight shift prerequisite** |
| **Pushes** | Enabled (gentle) | **Disabled** |
| **Knee Threshold** | 0.4m | **0.25m** |
| **Foot Clearance Weight** | 3.0 | **5.0** |
| **Weight Shift Reward** | None | **4.0 (new!)** |
| **Learning Path** | Direct to lifting | **Staged: shift → lift → sustain** |

## Curriculum (Future Enhancement)

Once robot masters basic lifting (80%+ episodes with foot clearance), enable:

1. **Phase 2**: Enable gentle pushes
   ```python
   self.events.push_robot = <gentle config>
   ```

2. **Phase 3**: Increase knee height threshold
   ```python
   params={"threshold": 0.35}  # then 0.4m
   ```

3. **Phase 4**: Increase push strength
   ```python
   velocity_range: (-0.3, 0.3)  # then (-0.5, 0.5)
   ```

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Summary

**v5 = The Complete Solution**

1. ✅ **Pushes disabled**: Stable learning environment
2. ✅ **Weight shift reward**: Teaches prerequisite skill
3. ✅ **Lower threshold**: Easier discovery (0.25m vs 0.4m)  
4. ✅ **High foot clearance**: Strong breakthrough signal (5.0)
5. ✅ **Staged learning**: Natural progression (shift → lift → sustain)
6. ✅ **Conditional rewards**: Prevents cheating strategies

The robot will learn single-leg balancing by **following the natural human approach**: shift weight first, then lift!


