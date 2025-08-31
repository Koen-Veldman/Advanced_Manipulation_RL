# play_analysis.py
# Copy of play.py with video recording, TensorBoard logging, and friction sensitivity analysis support

import os
import sys
import argparse
from datetime import datetime
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import your RL and Isaac Lab modules as in play.py
from isaaclab.app import AppLauncher
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
import Advanced_Manipulation_RL.tasks  # noqa: F401
import skrl
from packaging import version

# config shortcuts
algorithm = "ppo"
agent_cfg_entry_point = "skrl_cfg_entry_point"

parser = argparse.ArgumentParser(description="Play with a trained RL agent and log analysis.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to load.")
parser.add_argument("--friction_values", type=float, nargs='+', default=[1.0], help="List of friction values for sensitivity analysis.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for running the skrl agent.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.video = True
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

if version.parse(skrl.__version__) < version.parse("1.4.2"):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>=1.4.2'"
    )
    exit()
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_play_analysis"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    os.makedirs("videos/play_analysis", exist_ok=True)
    writer = SummaryWriter(log_dir)
    for friction in args_cli.friction_values:
        # Set friction in env config (example, adapt to your config structure)
        if hasattr(env_cfg.scene.cabinet.actuators["doors"], "static_friction"):
            env_cfg.scene.cabinet.actuators["doors"].static_friction = friction
        if hasattr(env_cfg.scene.cabinet.actuators["doors"], "dynamic_friction"):
            env_cfg.scene.cabinet.actuators["doors"].dynamic_friction = friction
        print(f"[INFO] Running evaluation with friction={friction}")
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join("videos", "play_analysis"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            env = gym.wrappers.RecordVideo(env, **video_kwargs)
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
        runner = Runner(env, agent_cfg)
        if resume_path:
            runner.agent.load(resume_path)
        # Run evaluation and collect rewards
        episode_rewards = []
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action = runner.agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards)
        print(f"[RESULT] Friction={friction}, Average Reward={avg_reward}")
        writer.add_scalar("eval/average_reward", avg_reward, friction)
        env.close()
    writer.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
