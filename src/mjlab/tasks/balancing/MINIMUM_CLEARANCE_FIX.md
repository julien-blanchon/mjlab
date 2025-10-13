# Minimum Clearance Fix - Ensuring Meaningful Foot Lift

## Issue Discovered

Your training showed:
- **`foot_height` reward**: ~2.0-2.2 (robot getting significant reward)
- **Visual observation**: Robot sitting with both feet on ground OR foot barely lifted

**Root cause:** The robot could get rewards from:
1. Small height differences between feet when both on slightly uneven ground
2. Contact sensor noise/timing issues
3. Barely lifting foot (5-10cm) and getting partial rewards

## The Fix: Minimum Absolute Clearance

### Added 5cm Minimum Clearance Check

Both `foot_height` and `foot_clearance` rewards now require:
```python
# CRITICAL: Foot must be lifted >5cm
min_clearance = 0.05  # 5cm absolute minimum
foot_actually_lifted = foot_height_above_ground > min_clearance

# Reward only given if BOTH:
# 1. Valid single-leg stance (contact sensors)
# 2. Foot actually lifted >5cm (meaningful lift)
both_conditions = valid_single_leg & foot_actually_lifted
return reward if both_conditions else 0.0
```

### Why 5cm Threshold?

- **<5cm**: Too small - could be from ground irregularities or contact sensor noise
- **5cm**: Clear lift - robot must intentionally raise foot
- **Purpose**: Prevents "micro-lifting" exploit where robot barely lifts to get partial rewards

## Updated Reward Structure

```python
POSITIVE REWARDS:
1. upright: 3.0              # Vertical torso (DOMINANT)
2. foot_height: 3.5          # INCREASED from 2.5 to push for higher lift
3. foot_clearance: 1.5       # Valid stance signal
4. static_stance: -0.5       # No horizontal movement
5. action_rate_l2: -0.01     # Smoothness

Total positive: 8.0
Total penalty: -0.51
```

### New Foot Height Parameters

```python
min_height: 0.10m (4 inches)    # Start rewarding earlier (was 0.15m)
optimal_height: 0.25m (10 inches) # Sweet spot (unchanged)
max_height: 0.40m (16 inches)   # Higher limit before penalty (was 0.35m)
```

**Effect:** Rewards now start at 10cm instead of 15cm, making it easier to discover lifting behavior.

## Reward Curve Comparison

| Foot Height | Old Reward | **New Reward** | Change |
|-------------|-----------|----------------|--------|
| 0.05m (2in) | 0.0 | 0.0 | Minimum clearance required |
| 0.10m (4in) | 0.0 | **0.3 × 3.5 = 1.05** | Start rewarding |
| 0.15m (6in) | 0.3 × 2.5 = 0.75 | **0.65 × 3.5 = 2.28** | +3.0x |
| **0.25m (10in)** | 1.0 × 2.5 = 2.5 | **1.0 × 3.5 = 3.5** | **+40% OPTIMAL** |
| 0.30m (12in) | 1.0 × 2.5 = 2.5 | **1.0 × 3.5 = 3.5** | Still optimal |
| 0.40m (16in) | Penalty | **0.3 × 3.5 = 1.05** | At limit |
| 0.50m (20in) | Strong penalty | **-0.75** | PENALTY! |

## Why This Will Work Better

### Before (Robot Stuck at 22cm)
```
Foot at 0.22m:
  upright: 3.0
  foot_height: 0.79 × 2.5 = 1.98 ← Current
  foot_clearance: 1.5
  static_stance: ~-0.05
  Total: ~6.43

Foot at 0.25m (target):
  upright: 3.0
  foot_height: 1.0 × 2.5 = 2.5
  foot_clearance: 1.5
  static_stance: ~-0.05
  Total: ~6.95

Difference: Only +0.52 reward to go from 22cm → 25cm (weak gradient!)
```

### After (Stronger Gradient)
```
Foot at 0.22m:
  upright: 3.0
  foot_height: 0.94 × 3.5 = 3.29
  foot_clearance: 1.5
  static_stance: ~-0.05
  Total: ~7.74

Foot at 0.25m (target):
  upright: 3.0
  foot_height: 1.0 × 3.5 = 3.5
  foot_clearance: 1.5
  static_stance: ~-0.05
  Total: ~7.95

Difference: +0.21 additional reward 
BUT higher absolute rewards make the gradient stronger in relative terms
```

## Expected Training Improvement

### Current Behavior (based on curves)
- Robot lifting to ~22cm and staying there
- foot_height saturating at ~2.0-2.2
- Not pushing to optimal 25cm

### Expected New Behavior
- Stronger reward signal for heights >22cm
- foot_height should grow to ~3.0-3.5
- Robot pushes toward 25-30cm optimal range

## Training Command

```bash
# Start fresh training with new rewards:
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096

# Or continue from checkpoint (rewards will shift):
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096 --agent.resume True --agent.load-run YOUR_RUN_ID
```

## Verification

**To verify the robot is actually lifting:**

1. **Check WandB video** at latest checkpoint
2. **Look for foot_height reward** approaching ~3.5 (means foot at 25-30cm)
3. **Episode mean reward** should approach ~8.0-8.5 (was ~6.5-7.0)

**Visual check:** Raised foot should be clearly visible, approximately knee-height off the ground (25-30cm).

## Summary

✅ **Added >5cm minimum clearance** to both foot rewards  
✅ **Increased foot_height weight**: 2.5 → 3.5 (stronger gradient)  
✅ **Adjusted height ranges**: Start at 10cm, optimal at 25-30cm  
✅ **Prevents micro-lifting**: Must lift >5cm to get ANY reward  

The robot should now push for the full 25-30cm foot lift instead of settling at 22cm!


