# Anti-Jumping Fix - Simple & Robust Solution

## The Problem

The robot was learning to **jump** (lift both feet off the ground) instead of standing on one leg. This happened because the reward structure had a critical loophole:

```python
# OLD foot_clearance (BROKEN):
raised_foot_off_ground = ~right_foot_contact  # Only checks raised foot
return raised_foot_off_ground.float()  # Gives reward even when jumping!
```

**The robot exploited this by:**
1. Jumping with both feet off ground → Gets `foot_clearance` reward ✓
2. Knees go up during jump → Gets `knee_height` reward ✓
3. Total reward ~7-10 without needing to balance!

## The Solution

### 1. Fixed Reward: `foot_clearance`

```python
# NEW foot_clearance (FIXED):
valid_single_leg = torch.where(
  standing_leg == 0,
  left_foot_contact & ~right_foot_contact,  # Standing foot MUST be on ground
  right_foot_contact & ~left_foot_contact,
)
return valid_single_leg.float()
```

**Now requires BOTH conditions:**
- ✓ Raised foot is OFF ground
- ✓ Standing foot is ON ground

**Jumping no longer gives reward!**

### 2. New Termination: `both_feet_off_ground`

```python
def both_feet_off_ground(env):
  """Terminate if both feet are off ground (jumping)."""
  both_off = ~left_foot_contact & ~right_foot_contact
  return both_off.float()
```

**Effect:** Episode immediately terminates if robot jumps.

### 3. Updated `balanced_stance_duration`

Now also requires standing foot to be on ground, consistent with the anti-jumping fix:

```python
in_balanced_stance = torch.where(
  standing_leg == 0,
  left_foot_contact & ~right_foot_contact,
  right_foot_contact & ~left_foot_contact,
)
```

## Simplified Reward Structure

Removed all the complex staged rewards and weight shift preparation. New structure:

```
PRIMARY REWARDS (8.1 total):
├── foot_clearance: 2.0          ← Fixed to prevent jumping
├── knee_height: 3.0              ← Exponential reward (saturates at threshold)
├── balanced_stance_duration: 2.0 ← Time-based reward for sustaining
├── upright: 1.0                  ← Maintain vertical orientation
└── alive: 0.1                    ← Small bonus for not falling

PENALTIES (-0.51 total, very light):
├── action_rate_l2: -0.01         ← Smoothness
└── joint_pos_limits: -0.5        ← Avoid joint limits
```

**Total positive rewards:** 8.1  
**Total penalties:** -0.51  
**Ratio:** 16:1 (encourages exploration)

## Why This Works

### Before (Jumping Exploit)
```
Jump (both feet off):
  foot_clearance: 5.0 ✓ (raised foot off ground)
  knee_height: ~3.0 ✓ (knees up during jump)
  upright: 2.0 ✓ (can stay upright while jumping)
  Total: ~10.0 reward

Risk: None! (jumping is easy)
```

### After (Anti-Jumping)
```
Jump (both feet off):
  foot_clearance: 0.0 ✗ (standing foot not on ground!)
  knee_height: 0.0 ✗ (episode terminated!)
  → Episode terminates immediately
  Total: 0.0 reward

Single-leg stance:
  foot_clearance: 2.0 ✓ (valid configuration)
  knee_height: 0-3.0 ✓ (based on height)
  balanced_stance_duration: 0-2.0 ✓ (grows with time)
  upright: 1.0 ✓
  Total: 5.1-8.1 reward (only way to succeed!)
```

## Other Key Changes

### Disabled Disturbances
- **Pushes:** Disabled (`push_robot = None`)
- **Foot friction randomization:** Disabled (`foot_friction = None`)

**Reason:** Robot needs a stable environment to discover balancing first. Can enable via curriculum later.

### Episode Length: 20 seconds
- Sufficient time to learn and maintain stance
- Not too long (40s was excessive for early training)

### Simpler Knee Height Reward
Uses exponential function that:
- Gives dense signal from 0m (not sparse!)
- Approaches 1.0 at threshold (0.4m)
- Saturates above threshold (doesn't encourage excessive height)

```python
normalized_height = relative_knee_height / threshold
reward = 1.0 - exp(-(normalized_height)^2 / (2*std^2))
```

## Expected Training Behavior

### Reward Curves
- `foot_clearance`: Should spike to ~2.0 when robot lifts foot (binary)
- `knee_height`: Gradual growth from 0 → ~2.0-3.0 (smooth exponential)
- `balanced_stance_duration`: Slow accumulation as robot learns to hold pose
- `upright`: Should stay near 1.0 throughout

### Terminations
- `both_feet_off_ground`: Should be HIGH initially (jumping attempts), then DROP to ~0
- `fell_over`: Moderate initially, decreases as learning progresses
- `body_contact_with_ground`: Should decrease over time
- `time_out`: Should become dominant termination mode

### Success Indicators
1. `both_feet_off_ground` terminations drop from high → near zero
2. `foot_clearance` reward becomes consistent ~2.0 (not sporadic)
3. `balanced_stance_duration` grows steadily (robot holding pose longer)
4. Episodes last closer to 20s (timeout) rather than early termination

## Training Command

```bash
MUJOCO_GL=egl uv run train Mjlab-Balancing-Flat-Unitree-G1 --env.scene.num-envs 4096
```

## Summary

**Core Fix:** Rewards now require standing foot to be ON ground, preventing jumping exploit.

**Philosophy:** Simple, robust rewards with clear anti-exploit mechanisms, rather than complex staged rewards that can be gamed.

**Result:** Robot will learn to stand on one leg (the only way to get rewards), rather than jumping (which now gives 0 reward and terminates episode).


