# Tree Pose Optimization - Classical Yoga Balance Position

## The Problem

The robot was balancing on one foot, but in an **extreme backbend position** with the raised leg way too high (~60cm+). This happened because:

1. **No penalty for excessive height**: Previous reward encouraged unlimited height
2. **Weak upright constraint**: upright weight was only 0.5, allowing extreme tilting
3. **No disturbances**: Robot didn't need to learn active balance recovery

**Result:** Robot found a stable but unnatural solution (backbend with leg very high).

## The Solution: Classical Tree Pose Rewards

### New Reward Hierarchy

```python
REWARDS (tree pose optimized):
1. upright: 3.0                    # DOMINANT - forces vertical orientation
2. knee_height_optimal_range: 2.5  # Optimal at 0.30-0.40m, penalizes >0.45m
3. foot_clearance: 1.5             # Progressive encouragement
4. action_rate_l2: -0.01           # Smoothness

Total positive: 7.0
Ratio: upright:height = 3.0:2.5 (upright is more important!)
```

### Why This Works

**Previous weights:**
- upright: 0.5
- knee_height: 5.0
- **Problem:** Robot could sacrifice upright (lose 0.5) to gain extra height (gain 2.5) → backbend optimal!

**New weights:**
- upright: 3.0  
- knee_height: 2.5
- **Solution:** Robot loses MORE from tilting (3.0) than it gains from extra height (0.5) → upright optimal!

## Knee Height Optimal Range Reward

### Reward Curve

| Height (m) | Reward | Description |
|------------|--------|-------------|
| 0.00 | 0.00 | Not lifting |
| 0.10 | 0.15 | Starting to lift |
| 0.20 | 0.30 | Minimum threshold |
| 0.25 | 0.65 | Good progress |
| **0.30** | **1.00** | **✓ OPTIMAL START** |
| **0.35** | **1.00** | **✓ OPTIMAL MIDDLE** |
| **0.40** | **1.00** | **✓ OPTIMAL END** |
| 0.42 | 0.86 | Getting high |
| 0.45 | 0.30 | At limit |
| **0.50** | **-0.15** | **PENALTY (too high!)** |
| **0.60** | **-0.45** | **STRONG PENALTY** |

The robot is **strongly penalized** for lifting knee above 0.45m!

### Comparison to Previous Behavior

**Old reward (unlimited height):**
```
Knee at 0.60m in extreme backbend:
  upright: ~0.3 × 0.5 = 0.15
  knee_height: ~1.5 × 5.0 = 7.5
  Total: ~7.65 ← Robot chose this!
```

**New reward (tree pose optimized):**
```
Knee at 0.60m in extreme backbend:
  upright: ~0.3 × 3.0 = 0.9
  knee_height: -0.45 × 2.5 = -1.125 (PENALTY!)
  Total: ~0.8 ← Robot will avoid!

Knee at 0.35m while upright:
  upright: ~1.0 × 3.0 = 3.0
  knee_height: 1.0 × 2.5 = 2.5
  Total: ~6.85 ← Robot will prefer!
```

## Gentle Push Disturbances

**Configuration:**
```python
Interval: 8-12 seconds (infrequent)
Linear velocity: ±0.2 m/s (gentle, not disruptive)
Angular velocity: ±0.1 rad/s (gentle rotation)
```

**Purpose:**
- Force robot to learn **active balance** (not just static poses)
- Teach **recovery strategies** when perturbed
- Prevent overfitting to perfect stillness

**Why gentle:**
- Robot needs time to establish stable stance first (8-12s)
- Small disturbances (±0.2 m/s) test balance without causing falls
- Builds robustness gradually

## Expected Training Behavior

### Reward Curves

- **`upright`** (3.0): Should saturate at ~3.0 (robot learns to stay vertical)
- **`knee_height`** (2.5): Should reach ~2.5 and stay there (0.30-0.40m range)
- **`foot_clearance`** (1.5): Should reach ~1.5 (consistent lifting)

### Visual Behavior

The robot should look like:
- ✅ Torso upright (vertical)
- ✅ Standing leg straight or slightly bent
- ✅ Raised knee at ~30-40cm height
- ✅ Arms may extend for balance (natural)
- ❌ NO extreme backbend
- ❌ NO leg raised super high (>45cm)

### Episode Progression

**Early training (0-2k steps):**
- Robot experiments with different poses
- Discovers that upright + moderate lift = best reward

**Mid training (2k-5k steps):**
- Robot converges to upright stance
- Knee height stabilizes around 0.30-0.35m
- Starts handling gentle pushes

**Late training (5k+ steps):**
- Consistent tree pose
- Quick recovery from pushes
- Episodes mostly timeout

## Comparison Table

| Feature | Old (Backbend) | **New (Tree Pose)** |
|---------|----------------|---------------------|
| **Upright weight** | 0.5 | **3.0** (6x increase!) |
| **Knee height weight** | 5.0 | **2.5** (halved) |
| **Max knee reward** | Unlimited | **Capped at 0.40m** |
| **Penalty >0.45m** | None | **-3.0 per meter** |
| **Pushes** | Disabled | **±0.2 m/s every 8-12s** |
| **Upright priority** | Secondary | **PRIMARY** |
| **Expected pose** | Backbend | **Vertical tree pose** |

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Success Metrics

After ~5k-10k steps, you should see:

✅ Robot stands **upright** (not tilted back)  
✅ Knee at **0.30-0.40m** height (not >0.50m)  
✅ Recovers from **gentle pushes**  
✅ Mean reward **~6-7** (not ~10 from extreme poses)  
✅ `upright` reward **~3.0** (saturated, robot is vertical)  
✅ `knee_height` reward **~2.5** (in optimal range)

## Summary

**The fix:**
1. ✅ Made `upright` DOMINANT (3.0 weight) - robot must stay vertical
2. ✅ Capped knee height at 0.40m with penalties above 0.45m
3. ✅ Added gentle pushes for active balance training
4. ✅ All major rewards conditional on valid single-leg stance

**Result:** Robot will learn the **classical tree pose** (upright, moderate knee lift) instead of extreme backbend!


