import argparse
import os
import tempfile
import time
from datetime import datetime

import numpy as np
import ray
import yaml
from experiments import torch_models  # noqa: F401
from foundation.utils.rllib_env_wrapper import RLlibEnvWrapper
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger


class MMOCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, **kwargs):
        episode.user_data["res"] = []

    def on_episode_step(self, *, worker, base_env, episode, **kwargs):
        info = episode.last_info_for("p")
        if info is None:
            return
        res = info.get("res")
        if res is not None:
            episode.user_data["res"].append(res)

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        results = episode.user_data.get("res", [])
        if not results:
            return
        res = np.array(results)
        idx = np.where(res[:, 0] > 0)[0]
        if idx.size == 0:
            return
        equality_segments = np.split(res[:, 1], idx + 1)
        equality_scores = [np.mean(seg) for seg in equality_segments if len(seg) > 0]
        equality = float(np.mean(equality_scores)) if equality_scores else 0.0
        episode.custom_metrics["profit"] = float(res[idx[-1], 0])
        episode.custom_metrics["equality"] = equality
        episode.custom_metrics["capability"] = float(res[idx[-1], 2])
        episode.custom_metrics["equXcap"] = float(np.mean(res[:, 1] * res[:, 2]))


def custom_log_creator(custom_path: str, custom_str: str):
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = f"{custom_str}_{timestr}"

    def logger_creator(config):
        os.makedirs(custom_path, exist_ok=True)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", type=str, default="", help="Adjustment type override")
    parser.add_argument("--restore", type=str, default="", help="Checkpoint path to restore")
    parser.add_argument("--cfg", type=str, default="", help="Experiment config name")
    parser.add_argument("--num-iter", type=int, default=-1, help="Number of training iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phase", type=int, default=1)
    return parser.parse_args()


def build_trainer(
    run_configuration: dict, env_config: dict, seed: int, logger_creator=None
):
    trainer_cfg = run_configuration.get("trainer", {})

    dummy_env = RLlibEnvWrapper(env_config, verbose=True)

    agent_policy_tuple = (
        None,
        dummy_env.observation_space,
        dummy_env.action_space,
        run_configuration.get("agent_policy"),
    )
    planner_policy_tuple = (
        None,
        dummy_env.observation_space_pl,
        dummy_env.action_space_pl,
        run_configuration.get("planner_policy"),
    )
    policies = {"a": agent_policy_tuple, "p": planner_policy_tuple}

    def policy_mapping_fun(agent_id):
        return "a" if str(agent_id).isdigit() else "p"

    if run_configuration["general"].get("train_planner"):
        policies_to_train = ["a", "p"]
    else:
        policies_to_train = ["a"]

    metrics_smoothing = trainer_cfg.get("metrics_smoothing_episodes")
    if metrics_smoothing is None:
        metrics_smoothing = (
            trainer_cfg.get("num_workers", 0) * trainer_cfg.get("num_envs_per_worker", 1)
        )

    config_builder = (
        PPOConfig()
        .environment(env=RLlibEnvWrapper, env_config=env_config)
        .framework("torch")
        .seed(seed)
        .rollouts(
            num_rollout_workers=trainer_cfg.get("num_workers", 0),
            num_envs_per_worker=trainer_cfg.get("num_envs_per_worker", 1),
            rollout_fragment_length=trainer_cfg.get("rollout_fragment_length", 200),
            batch_mode=trainer_cfg.get("batch_mode", "truncate_episodes"),
            observation_filter=trainer_cfg.get("observation_filter", "NoFilter"),
        )
        .training(
            train_batch_size=trainer_cfg.get("train_batch_size", 4000),
            sgd_minibatch_size=trainer_cfg.get("sgd_minibatch_size", 128),
            num_sgd_iter=trainer_cfg.get("num_sgd_iter", 10),
            shuffle_sequences=trainer_cfg.get("shuffle_sequences", True),
        )
        .resources(num_gpus=trainer_cfg.get("num_gpus", 0))
        .reporting(metrics_num_episodes_for_smoothing=metrics_smoothing)
        .multi_agent(
            policies=policies,
            policies_to_train=policies_to_train,
            policy_mapping_fn=policy_mapping_fun,
        )
        .callbacks(MMOCallbacks)
    )

    return config_builder.build(logger_creator=logger_creator)


def load_run_configuration(cfg_path: str, phase: int, adj: str) -> tuple[dict, dict]:
    config_path = os.path.join("./experiments", f"{cfg_path}.yaml")
    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    if adj:
        run_configuration["env"]["adjustemt_type"] = adj

    run_configuration["general"]["train_planner"] = phase == 2

    trainer_config = run_configuration.get("trainer", {})
    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker", 1),
    }
    return run_configuration, env_config


def train(trainer, num_iters: int, save_dir: str, save_metric: str):
    os.makedirs(save_dir, exist_ok=True)
    trainer.save(save_dir)
    metric_log = []
    best_metric = -np.inf
    start_time = time.time()

    for iteration in range(num_iters):
        print(f"********** Iter : {iteration} **********")
        result = trainer.train()
        current_time = time.time()

        policy_rewards = result.get("policy_reward_mean", {})
        if save_metric in policy_rewards:
            metric_value = policy_rewards[save_metric]
            if metric_value > best_metric:
                best_metric = metric_value
                trainer.save(os.path.join(save_dir, f"rew_{metric_value:.4f}"))

            iter_time = round(current_time - start_time, 4)
            episode_reward_mean = round(result.get("episode_reward_mean", 0.0), 6)
            a_rew = round(policy_rewards.get("a", 0.0), 6)
            p_rew = round(policy_rewards.get("p", 0.0), 6)

            custom_metrics = result.get("custom_metrics", {})
            profit = round(custom_metrics.get("profit_mean", 0.0), 6)
            equality = round(custom_metrics.get("equality_mean", 0.0), 6)
            capability = round(custom_metrics.get("capability_mean", 0.0), 6)
            equxcap = round(custom_metrics.get("equXcap_mean", 0.0), 6)

            print(
                "time: {} epi_rew: {} a_rew: {} p_rew: {} epi_len: {} pro: {} equ: {} cap: {} prod: {}".format(
                    iter_time,
                    episode_reward_mean,
                    a_rew,
                    p_rew,
                    result.get("episode_len_mean", 0.0),
                    profit,
                    equality,
                    capability,
                    equxcap,
                )
            )

            metric_log.append(
                {
                    "iter": iteration,
                    "epi_len": result.get("episode_len_mean", 0.0),
                    "epi_rew": episode_reward_mean,
                    "a_rew": a_rew,
                    "p_rew": p_rew,
                    "profit": profit,
                    "equlity": equality,
                    "capability": capability,
                    "equXcap": equxcap,
                }
            )
            start_time = current_time
        else:
            print(f"episode_reward_mean: {result.get('episode_reward_mean', 0.0)}")

        if iteration % 10 == 9:
            trainer.save(os.path.join(save_dir, f"iter_{iteration}"))
            print(f"save ckpt at iter {iteration}")

    np.save(os.path.join(save_dir, "metric.npy"), metric_log)


def main():
    args = parse_args()
    run_configuration, env_config = load_run_configuration(args.cfg, args.phase, args.adj)

    ray.init(ignore_reinit_error=True)

    save_dir = os.path.join(
        "runs",
        f"phase_{args.phase}_{run_configuration['env']['adjustemt_type']}_seed_{args.seed}",
        args.cfg[:12],
    )

    logger_creator = custom_log_creator(save_dir, "train")

    trainer = build_trainer(
        run_configuration, env_config, args.seed, logger_creator=logger_creator
    )

    if args.restore:
        trainer.restore(args.restore)

    num_iters = 400 if args.num_iter == -1 else args.num_iter
    save_metric = "a" if args.phase == 1 else "p"
    train(trainer, num_iters, save_dir, save_metric)


if __name__ == "__main__":
    main()
