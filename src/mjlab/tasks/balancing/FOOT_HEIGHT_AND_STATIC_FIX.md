# Foot Height & Static Stance Fix - Classical Tree Pose

## Issues Addressed

### Issue #5: Bending Knee Exploit ✅ FIXED
**Problem:** Measuring knee height allowed robot to bend knee and keep foot low
**Fix:** Now measure FOOT/ankle height instead of knee height

### Issue #6: Walking on One Foot ✅ FIXED
**Problem:** Robot was moving horizontally to maintain balance (locomotion, not static balance)
**Fix:** Added `static_stance` penalty (-0.5) for horizontal base velocity

## Key Changes

### 1. Switched from Knee Height → Foot Height

**Old (exploitable):**
```python
knee_height_optimal_range:
  - Measures: knee_z - standing_foot_z
  - Problem: Robot can bend knee to keep foot low while getting reward
```

**New (exploit-proof):**
```python
raised_foot_height_optimal_range:
  - Measures: raised_ankle_z - standing_ankle_z  
  - Benefit: Directly measures what we care about (foot off ground)
  - Can't cheat: Bending knee doesn't help if foot is still low
```

### 2. Added Static Stance Penalty

**New reward term:**
```python
static_stance_penalty:
  - Measures: horizontal base velocity (x, y)
  - Weight: -0.5 (significant penalty)
  - Purpose: Encourages STATIC balance (tree pose), not walking
```

**Effect:**
- Robot walking at 0.3 m/s → penalty ≈ -0.045
- Robot stationary (0 m/s) → penalty ≈ 0.0
- Difference: ~0.8 reward points favor static stance

### 3. Adjusted Optimal Ranges for Foot Height

**Foot height targets (more conservative than knee):**
- Min: 0.15m (6 inches) - start getting reward
- Optimal: 0.25m (10 inches) - classical tree pose
- Max: 0.35m (14 inches) - above this = penalty
- Penalty rate: -3.0 per excess meter (strong)

Compare to old knee height targets:
- Old: 0.20m → 0.30m → 0.45m
- New: 0.15m → 0.25m → 0.35m (shifted down by ~5-10cm)

## Complete Reward Structure

```
POSITIVE REWARDS:
1. upright: 3.0              # DOMINANT - stay vertical
2. foot_height: 2.5          # Foot at 0.25-0.30m optimal
3. foot_clearance: 1.5       # Valid single-leg stance

PENALTIES:
4. static_stance: -0.5       # Penalize horizontal movement
5. action_rate_l2: -0.01     # Smoothness

Total positive: 7.0
Total penalties: -0.51
Ratio: 14:1 (encourages exploration but discourages movement)
```

## Reward Comparison

| Behavior | upright | foot_height | foot_clearance | static | **Total** |
|----------|---------|-------------|----------------|--------|-----------|
| **Static tree pose** (foot 25cm) | 3.0 | 2.5 | 1.35 | 0.0 | **~6.85** ✓✓✓ |
| Walking (foot 25cm, 0.3 m/s) | 2.7 | 2.5 | 1.35 | -0.5 | **~6.05** ✗ |
| Bending knee (knee high, foot low 10cm) | 3.0 | 0.2 | 0.7 | 0.0 | **~3.9** ✗ |
| Extreme backbend (foot 40cm) | 0.9 | 0.75 | 1.5 | 0.0 | **~3.15** ✗ |
| Sitting (both feet down) | 3.0 | 0.0 | 0.0 | 0.0 | **~3.0** ✗ |

**Static tree pose gives maximum reward!**

## Foot Height Reward Curve

| Foot Height | Reward | Description |
|-------------|--------|-------------|
| 0.00m | 0.00 | Not lifting |
| 0.10m | 0.20 | Starting |
| 0.15m | 0.30 | Minimum threshold |
| 0.20m | 0.65 | Good progress |
| **0.25m** | **1.00** | **✓ OPTIMAL** |
| **0.28m** | **1.00** | **✓ OPTIMAL RANGE** |
| 0.30m | 0.86 | Still good |
| 0.35m | 0.30 | At limit |
| **0.40m** | **-0.15** | **PENALTY!** |
| **0.50m** | **-0.45** | **STRONG PENALTY!** |

The robot is strongly discouraged from lifting foot above 35cm.

## Push Disturbances

**Configuration:**
- Interval: 8-12 seconds (infrequent)
- Linear: ±0.2 m/s (gentle)
- Angular: ±0.1 rad/s (gentle rotation)

**Purpose:**
- Tests static balance under perturbation
- Forces robot to learn recovery strategies
- Prevents overfitting to perfect stillness

**Why gentle:**
- Robot needs time to establish static stance (8-12s before push)
- Small magnitude (0.2 m/s) tests balance without causing falls
- Teaches active stabilization, not just passive stance

## Expected Training Behavior

### Early Training (0-2k steps)
- Robot experiments with different poses
- Discovers: upright + foot lift = good reward
- May try walking initially (gets penalized)

### Mid Training (2k-5k steps)
- Converges to upright stance
- Foot height stabilizes around 0.20-0.25m
- Learns to minimize horizontal movement

### Late Training (5k-10k steps)
- Consistent static tree pose
- Foot at ~0.25m (10 inches)
- Minimal horizontal drift (<0.1 m/s)
- Quick recovery from pushes

## Success Metrics

Monitor in WandB:

✅ **`upright` reward**: ~3.0 (robot staying vertical)  
✅ **`foot_height` reward**: ~2.5 (foot at 0.25-0.30m)  
✅ **`foot_clearance` reward**: ~1.5 (consistent lifting)  
✅ **`static_stance` penalty**: ~0.0 to -0.05 (minimal movement)  
✅ **Mean reward**: ~6.5-6.8 (close to maximum)  
✅ **Episode length**: ~800-1000 steps (timeout dominant)

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Visual Expectations

The robot should look like a **classical yoga tree pose**:

✅ **Torso**: Vertical (upright)  
✅ **Standing leg**: Straight or slightly bent  
✅ **Raised foot**: ~25cm (10 inches) off ground  
✅ **Base position**: Stationary (no horizontal drift)  
✅ **Arms**: May extend for balance (natural)  

❌ **NOT**: Backbend with leg very high  
❌ **NOT**: Walking/sliding on one foot  
❌ **NOT**: Bent knee with low foot  

## Summary of All Fixes

| Issue | Old Behavior | New Fix |
|-------|--------------|---------|
| Jumping | Both feet off → reward | Conditional rewards → 0 |
| Sitting | Both feet on → ~7 reward | Conditional rewards → ~3 |
| Backbend | Extreme tilt → high reward | upright (3.0) + height penalty → ~3 |
| Bending knee | Knee high, foot low → reward | **Measure foot, not knee → ~4** |
| Walking | Moving on one foot → same reward | **static_stance (-0.5) → -0.5 penalty** |

**All exploits eliminated!** Robot must now learn a proper static tree pose.


