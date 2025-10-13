"""Event terms for single-leg balancing task."""

import torch

from mjlab.envs import ManagerBasedEnv


def randomize_standing_leg(
  env: ManagerBasedEnv,
  env_ids: torch.Tensor,
) -> None:
  """Randomly select which leg to stand on at the start of each episode.
  
  Randomly chooses left (0) or right (1) leg as the standing leg.
  The choice is stored in env.extras["standing_leg_choice"].
  Also resets the balanced stance duration counter.
  
  Args:
    env: The environment instance.
    env_ids: Environment indices for which to randomize the standing leg.
  """
  # Initialize standing_leg_choice if not present
  if "standing_leg_choice" not in env.extras:
    env.extras["standing_leg_choice"] = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.long
    )
  
  # Initialize balanced_stance_counter if not present
  if "balanced_stance_counter" not in env.extras:
    env.extras["balanced_stance_counter"] = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
  
  # Randomly choose 0 (left) or 1 (right) for each environment being reset
  random_choice = torch.randint(0, 2, (len(env_ids),), device=env.device, dtype=torch.long)
  env.extras["standing_leg_choice"][env_ids] = random_choice
  
  # Reset balanced stance counter for reset environments
  env.extras["balanced_stance_counter"][env_ids] = 0.0

