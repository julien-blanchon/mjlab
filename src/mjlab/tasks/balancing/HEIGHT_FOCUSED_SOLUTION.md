# Height-Focused Solution - Encouraging Strong Knee Lift

## Problem

Robot was learning to **barely lift the foot** (1mm off ground) instead of raising knee to target height (0.4m).

**Training curves showed:**
- `foot_clearance`: ~1.2-1.4 → Foot clearing ground intermittently
- `knee_height`: ~1.5-1.8 → Some height but not progressing
- `upright`: ~0.8-0.9 → Robot prioritizing balance over height

**Root cause:** Rewards weren't strong enough to push for significant height gain.

## The Fix: Make Knee Height DOMINANT

### New Reward Structure

```python
REWARDS (ordered by importance):
1. knee_height: 5.0         # DOMINANT - lifting is the primary objective  
2. foot_clearance: 2.0      # Binary signal (standing foot ON, raised foot OFF)
3. upright: 0.5             # Reduced - allow some tilt to achieve lift
4. action_rate_l2: -0.01    # Light smoothness penalty

Total: 7.5 positive, -0.01 penalty
```

### Why This Works

**Old weights:**
- knee_height: 2.0
- foot_clearance: 3.0  
- upright: 1.0
- **Problem:** Robot could get 4.0 reward (clearance + upright) without lifting knee high

**New weights:**
- knee_height: 5.0
- foot_clearance: 2.0
- upright: 0.5
- **Solution:** Maximum reward (7.5) only achievable by lifting knee to 0.4m+

## Improved Knee Height Reward Function

**Old:** Exponential reward (sparse at low heights)
```python
reward = 1.0 - exp(-(h/threshold)^2 / (2*std^2))
```

**New:** Linear + bonus (dense feedback at all heights)
```python
# Linear component: 0.0 → 1.0 from height 0 → threshold
linear_reward = clamp(height / 0.4, 0.0, 1.0)

# Bonus for exceeding threshold
bonus = clamp((height - 0.4) / 0.08, 0.0, 0.5)

# Total: 0.0 → 1.5
return linear_reward + bonus
```

### Reward Curve Comparison

| Height (m) | Old (exp) | **New (linear+bonus)** | With 5.0 weight |
|------------|-----------|------------------------|-----------------|
| 0.00       | 0.00      | **0.00**               | **0.0**         |
| 0.05       | ~0.05     | **0.125**              | **0.63**        |
| 0.10       | ~0.18     | **0.25**               | **1.25**        |
| 0.20       | ~0.55     | **0.50**               | **2.50**        |
| 0.30       | ~0.80     | **0.75**               | **3.75**        |
| **0.40**   | **0.93**  | **1.00**               | **5.00** ✓      |
| **0.48**   | **0.98**  | **1.50**               | **7.50** ✓✓     |

The new reward gives **much stronger feedback** at intermediate heights!

## Expected Training Behavior

### Reward Curves

- **`knee_height`** (5.0): Should dominate total reward, growing from 0 → 5.0+
- **`foot_clearance`** (2.0): Should saturate at ~2.0 when robot consistently lifts
- **`upright`** (0.5): Lower priority, ~0.4-0.5

### Episode Progression

**Early (0-1k steps):**
- knee_height: 0.0 → 1.0 (robot discovers lifting gives reward)
- Mean reward: ~0.5 → ~3.0

**Mid (1k-3k steps):**
- knee_height: 1.0 → 3.0 (progressive height gain)
- Mean reward: ~3.0 → ~6.0

**Late (3k+ steps):**
- knee_height: 3.0 → 5.0 (approaching 0.4m threshold)
- foot_clearance: → 2.0 (consistent single-leg stance)
- Mean reward: ~6.0 → ~7.5

### Success Metrics

✅ Mean reward reaches ~6-7  
✅ `knee_height` reward >4.0 (knee at ~0.35-0.40m)  
✅ `foot_clearance` stays at ~2.0 (consistent)  
✅ Episodes timeout (not fall over)

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Summary of Changes

| Change | Old | New | Why |
|--------|-----|-----|-----|
| **knee_height weight** | 2.0 | **5.0** | Make height the primary objective |
| **knee_height function** | Exponential | **Linear+bonus** | Dense feedback at all heights |
| **upright weight** | 1.0 | **0.5** | Allow robot to tilt slightly for higher lift |
| **foot_clearance weight** | 3.0 | **2.0** | Still important, but secondary |

## Why This Will Work

1. **Dominant height signal**: 5.0 weight means every cm of height is worth MORE than clearance or upright
2. **Linear rewards**: Robot gets immediate feedback for ANY height increase (not sparse)
3. **Progressive optimization**: 
   - First learns to clear foot (small reward)
   - Then learns to lift higher (growing reward)
   - Finally reaches 0.4m threshold (maximum reward)
4. **Reduced upright constraint**: Robot can tilt slightly if needed to achieve higher lift

The robot will now optimize for **maximum knee height** as the primary objective!


