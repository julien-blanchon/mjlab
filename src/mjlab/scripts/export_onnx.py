"""Export a trained policy to ONNX format with normalization.

This script loads a trained checkpoint and exports it to ONNX format with
observation normalization properly embedded.

Usage:
    # Export from wandb checkpoint
    mjpython -m mjlab.scripts.export_onnx \\
        --task Mjlab-Balancing-Flat-Unitree-G1-Play \\
        --wandb-run-path blanchon/mjlab/runs/twd0ip46 \\
        --output-dir ./public/examples/checkpoints/unitree_g1 \\
        --output-name single_foot_balance.onnx

    # Export from local checkpoint
    mjpython -m mjlab.scripts.export_onnx \\
        --task Mjlab-Balancing-Flat-Unitree-G1-Play \\
        --checkpoint-file ./path/to/model.pt \\
        --output-dir ./public/examples/checkpoints/unitree_g1 \\
        --output-name single_foot_balance.onnx
"""

import os
from pathlib import Path
from typing import Literal, cast

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.balancing.balancing_env_cfg import BalancingEnvCfg
from mjlab.tasks.balancing.rl.exporter import export_balancing_policy_as_onnx
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.tasks.velocity.rl.exporter import export_velocity_policy_as_onnx
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
  load_cfg_from_registry,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


def export_policy(
  task: str,
  wandb_run_path: str | None = None,
  checkpoint_file: str | None = None,
  output_dir: str = "./exported_policies",
  output_name: str = "policy.onnx",
  device: str = "cpu",
):
  """Export a trained policy to ONNX format.
  
  Args:
    task: Task name (e.g., 'Mjlab-Balancing-Flat-Unitree-G1-Play')
    wandb_run_path: W&B run path (e.g., 'blanchon/mjlab/runs/twd0ip46')
    checkpoint_file: Local checkpoint file path
    output_dir: Directory to save the ONNX model
    output_name: Output filename (e.g., 'policy.onnx')
    device: Device to load the model on
  """
  configure_torch_backends()

  print(f"[INFO]: Using device: {device}")
  print(f"[INFO]: Task: {task}")

  # Load configs
  env_cfg = cast(
    ManagerBasedRlEnvCfg, load_cfg_from_registry(task, "env_cfg_entry_point")
  )
  agent_cfg = cast(
    RslRlOnPolicyRunnerCfg, load_cfg_from_registry(task, "rl_cfg_entry_point")
  )

  # Determine checkpoint path
  if checkpoint_file is not None:
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"[INFO]: Loading checkpoint: {checkpoint_path}")
  elif wandb_run_path is not None:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    checkpoint_path = get_wandb_checkpoint_path(log_root_path, Path(wandb_run_path))
    print(f"[INFO]: Loading W&B checkpoint: {checkpoint_path}")
  else:
    raise ValueError("Must provide either --wandb-run-path or --checkpoint-file")

  # Create a minimal runner just to load the policy
  from dataclasses import asdict
  import gymnasium as gym
  from mjlab.rl import RslRlVecEnvWrapper
  
  # Create env (we won't use it, just need it for runner init)
  env = gym.make(task, cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  
  # Create runner
  agent_cfg_dict = asdict(agent_cfg)
  runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=".", device=device)
  
  # Load checkpoint
  runner.load(str(checkpoint_path), map_location=device)
  
  # Get the actor-critic
  actor_critic = runner.alg.policy
  
  # Get normalizer if present
  if actor_critic.actor_obs_normalization:
    normalizer = actor_critic.actor_obs_normalizer
    print(f"[INFO]: Using observation normalizer")
    print(f"  - Mean shape: {normalizer.mean.shape}")
    print(f"  - Std shape: {normalizer.std.shape}")
  else:
    normalizer = None
    print(f"[WARNING]: No observation normalization found in checkpoint!")
  
  # Create output directory
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  
  # Export based on task type
  print(f"[INFO]: Exporting to ONNX: {output_path / output_name}")
  
  if isinstance(env_cfg, BalancingEnvCfg):
    export_balancing_policy_as_onnx(
      actor_critic=actor_critic,
      path=str(output_path),
      normalizer=normalizer,
      filename=output_name,
      verbose=True,
    )
  else:
    # Fallback to velocity exporter for other tasks
    export_velocity_policy_as_onnx(
      actor_critic=actor_critic,
      path=str(output_path),
      normalizer=normalizer,
      filename=output_name,
      verbose=True,
    )
  
  print(f"[SUCCESS]: ONNX model exported to {output_path / output_name}")
  
  # Clean up
  env.close()


def main():
  """Entry point for the CLI."""
  tyro.cli(export_policy)


if __name__ == "__main__":
  main()

