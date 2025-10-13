# Conditional Rewards Fix - Preventing the Sitting Exploit

## The Bug You Found

Looking at your training curves:
- **`knee_height`**: ~7.0 (maxed out!)
- **`foot_clearance`**: ~1.8
- **Robot in video**: Sitting with both feet on ground

**The robot was getting MASSIVE rewards (~7.0) without actually lifting!**

## Root Cause: Natural Knee-Ankle Geometry

When the robot sits with both feet on the ground, the knee is naturally ~0.4-0.5m above the ankle (just the leg geometry). The old `knee_height` function was rewarding this natural distance, even when not in single-leg stance!

```python
# OLD (BROKEN):
relative_knee_height = raised_knee_height - standing_foot_height
reward = linear_reward(relative_knee_height)  # Gets reward even when sitting!
return reward × 5.0 weight = ~7.0 reward for doing nothing!
```

## The Fix: Conditional Rewards

Both `knee_height` and `foot_clearance` now **ONLY give rewards** when in valid single-leg stance:
- Standing foot is ON ground
- Raised foot is OFF ground

### Fixed `knee_height` Reward

```python
def knee_height_above_threshold(env, threshold, std):
    # Check if in valid single-leg stance
    valid_single_leg = torch.where(
        standing_leg == 0,
        left_foot_contact & ~right_foot_contact,
        right_foot_contact & ~left_foot_contact,
    )
    
    # Calculate knee height (same as before)
    relative_knee_height = raised_knee_height - standing_foot_height
    reward = linear_reward(relative_knee_height)
    
    # ONLY return reward if in valid stance!
    return torch.where(
        valid_single_leg,
        reward,
        torch.zeros_like(reward),  # 0 reward if both feet down!
    )
```

### Enhanced `foot_clearance` Reward

Now also measures how high the foot is lifted (not just binary on/off):

```python
def foot_clearance(env):
    # Check valid single-leg stance
    valid_single_leg = (standing_foot_on & raised_foot_off)
    
    # Measure raised foot height above ground
    foot_height_above_ground = raised_ankle_height - standing_ankle_height
    
    # Progressive reward: 0.5→1.0 from 0cm→15cm
    height_reward = 0.5 + 0.5 * clamp(foot_height / 0.15, 0, 1)
    
    # Only give if in valid stance
    return height_reward if valid_single_leg else 0.0
```

This encourages CLEAR foot lifting (15cm+), not just barely touching off ground.

## Reward Structure Now

```python
REWARDS (all conditional on valid single-leg stance):
1. knee_height: 5.0          # 0.0 if both feet on ground!
2. foot_clearance: 2.0        # 0.0 if both feet on ground!
3. upright: 0.5              # Always active (small)
4. action_rate_l2: -0.01     # Always active (tiny penalty)
```

## Behavior Comparison

### Before (Sitting Exploit)
```
Robot sitting with both feet on ground:
  knee_height: ~7.0 ✓ (natural leg geometry!)
  foot_clearance: 0.0
  upright: ~0.5
  Total: ~7.5 reward for doing NOTHING!
```

### After (Conditional Rewards)
```
Robot sitting with both feet on ground:
  knee_height: 0.0 ✗ (no valid single-leg stance)
  foot_clearance: 0.0 ✗ (no valid single-leg stance)
  upright: ~0.5 ✓
  Total: ~0.5 reward (must actually lift to get more!)

Robot with valid single-leg stance:
  knee_height: 0-7.5 ✓ (based on actual height)
  foot_clearance: 1.0-2.0 ✓ (based on foot height)
  upright: ~0.5 ✓
  Total: ~2-10 reward (only way to succeed!)
```

## Expected Training Behavior

### Reward Curves

- **`knee_height`** (5.0): Should start at ~0, then grow to 5-7.5 as robot learns to lift
- **`foot_clearance`** (2.0): Should start at ~0, then reach 1-2 when robot lifts consistently
- **`upright`** (0.5): Should stay around 0.4-0.5 throughout

### Episode Length

- Should remain high (~500-1000 steps)
- Most terminations should be `time_out`, not `fell_over`

### Mean Reward

- Early: ~0.5 (just upright, both feet down)
- Mid: ~3-5 (robot discovering lifting)
- Late: ~8-10 (robot mastering single-leg stance with knee at 0.4m)

## Summary

✅ **Sitting exploit FIXED**: Robot gets ~0.5 reward (not ~7.5) when sitting  
✅ **Must lift to succeed**: All major rewards require valid single-leg stance  
✅ **Progressive encouragement**: foot_clearance rewards higher lifts (not just barely off ground)  
✅ **Clear gradient**: From ~0.5 (sitting) → ~10.0 (perfect stance)

The robot will now learn that it MUST lift one foot to get rewards!


