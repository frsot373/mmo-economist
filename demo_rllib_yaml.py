import os

import numpy as np
import ray
import yaml
from experiments import torch_models  # noqa: F401
from foundation.utils.rllib_env_wrapper import RLlibEnvWrapper
from ray.rllib.algorithms.callbacks import DefaultCallbacks

try:  # pragma: no cover - compatibility with older Ray versions
    from ray.rllib.algorithms.ppo import PPO as PPOTrainer
except ImportError:  # pragma: no cover
    from ray.rllib.agents.ppo import PPOTrainer

ray.init(ignore_reinit_error=True)


class DemoCallbacks(DefaultCallbacks):
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
        episode.custom_metrics["profit"] = float(np.mean(res[idx, 0]))
        episode.custom_metrics["equality"] = equality
        episode.custom_metrics["capability"] = float(np.mean(res[idx, 2]))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adj', type=str, default='random-asy')# adjust type
    parser.add_argument('--restore', type=str, default='')# restore ckpt path
    parser.add_argument('--cfg', type=str, default='config_50_50')# config_50_50.yaml
    parser.add_argument('--num-iter', type=int, default=2)# train iter
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--phase', type=int, default=1)# 1/2
    res =parser.parse_args()
    return res

def init_trainer(args):


    #config_path = os.path.join('./experiments', "config_50_50.yaml")
    # cfg_path='config_20_80.yaml'
    cfg_path=args.cfg
    config_path = os.path.join('./experiments', cfg_path+'.yaml')

    with open(config_path, "r") as f:
        run_configuration = yaml.safe_load(f)

    trainer_config = run_configuration.get("trainer")


    if args.adj!='':
        run_configuration["env"]["adjustemt_type"]=args.adj

    env_config = {
        "env_config_dict": run_configuration.get("env"),
        "num_envs_per_worker": trainer_config.get("num_envs_per_worker"),
    }

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
    def policy_mapping_fun(i): return "a" if str(i).isdigit() else "p"


    if run_configuration["general"]["train_planner"]:
        policies_to_train = ["a", "p"]
    else:
        policies_to_train = ["a"]

    trainer_config.update(
        {
            "env_config": env_config,
            "seed": args.seed,
            "multiagent": {
                "policies": policies,
                "policies_to_train": policies_to_train,
                "policy_mapping_fn": policy_mapping_fun,
            },
            "metrics_smoothing_episodes": trainer_config.get("num_workers")
            * trainer_config.get("num_envs_per_worker"),
        }
    )

    trainer_config['callbacks'] = DemoCallbacks

    exp_dir='runs/'
    log_dir=f"/phase_{args.phase}_{run_configuration['env']['adjustemt_type']}_{cfg_path[:9]}"
    save_dir=f"runs/phase_{args.phase}_{run_configuration['env']['adjustemt_type']}/{cfg_path[:9]}"

    trainer = PPOTrainer(env=RLlibEnvWrapper, config=trainer_config)

    return trainer,save_dir

def train(trainer,args,save_dir):
    NUM_ITERS = 400 if args.num_iter==-1 else args.num_iter
    save_metric='a' if args.phase==1 else 'p'
    metric_log=[]
    cur_best=0
    import time
    pst_time=time.time()
    trainer.save(save_dir)
    for iteration in range(NUM_ITERS):
        print(f'********** Iter : {iteration} **********')
        result = trainer.train()
        # import ipdb;ipdb.set_trace()
        #r1 = trainer.workers.local_worker().sampler.get_data().policy_batches
        #print(r1['a']['rewards'].sum(), result['episode_reward_mean'])

        cur_time = time.time()

        if save_metric in result['policy_reward_mean'].keys():
            if result['policy_reward_mean'][save_metric]>cur_best:
                cur_best=result['policy_reward_mean'][save_metric]
                trainer.save(f"{save_dir}/rew_{round(cur_best,4)}")
            iter_time=round(cur_time-pst_time,4)
            episode_reward_mean=round(result.get('episode_reward_mean'),4)
            a_rew=round(result['policy_reward_mean']['a'],4)
            p_rew=round(result['policy_reward_mean']['p'],4)

            if 'profit_mean' in result['custom_metrics'].keys():
                profit=round(result['custom_metrics']['profit_mean'],4)
                equality=round(result['custom_metrics']['equality_mean'],4)
                capability=round(result['custom_metrics']['capability_mean'],4)
                print(f"time: {iter_time} epi_rew: {episode_reward_mean} a_rew:{a_rew} ",
                      f" p_rew:{p_rew}, epi_len: {result['episode_len_mean'] }",
                      f" pro:{profit} equ:{equality} cap:{capability} ")
                metric={'iter':iteration,'epi_rew':episode_reward_mean,
                        'a_rew':a_rew,'p_rew':p_rew,
                        'profit':profit,'equlity':equality,
                        'capability':capability,
                        }
                metric_log.append(metric)
            pst_time=cur_time
        else:
            print(f"episode_reward_mean: {result.get('episode_reward_mean')}")
        if iteration  % 10 == 9 :
            trainer.save(f"{save_dir}/iter_{iteration}")
            print(f"save ckpt at iter {iteration}")
    np.save(os.path.join(save_dir,'metric.npy'),metric_log)

if __name__=="__main__":
    args=parse_args()
    trainer,save_dir=init_trainer(args)
    if args.restore!='':
        trainer.restore(args.restore)
        # trainer.restore('dir_ckpt_random-asy/iter_399/checkpoint_400/checkpoint-400')
    train(trainer,args,save_dir)
