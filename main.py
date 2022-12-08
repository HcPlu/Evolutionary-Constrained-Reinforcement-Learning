# -*- coding:utf-8 _*-

import copy

import numpy as np, os, time, random
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,'
from envs_repo.constructor import EnvConstructor,SafeEnvConstructor
from models.constructor import ModelConstructor
from core.params import Parameters
import argparse, torch
from algos.erl_trainer import ERL_Trainer
import argparse
import datetime
import os
import pprint

import torch
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from algos.rcpoSACPolicy import rcpoSACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic

class actor_constructor():
    def __init__(self,state_shape,action_shape,hidden_sizes,max_action,policy):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.hidden_sizes = hidden_sizes
        self.max_action = max_action
        self.policy = policy


    def build_actor(self,device):
        net_a = Net(self.state_shape, hidden_sizes=self.hidden_sizes,device=device)
        actor = ActorProb(
            net_a,
            self.action_shape,
            max_action=self.max_action,
            unbounded=True,
            device=device,
            conditioned_sigma=True,
        )

        return actor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='Env Name',  default='Pendulum-v0')
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--auto-alpha", default=False, action="store_true")
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--update-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--render", type=float, default=0.)
    parser.add_argument("--rcpo_lambda",type = float,default=0.0001)
    parser.add_argument("--rcpo_lr",type = float,default=0.0001)
    parser.add_argument("--rcpo_alpha", type=float, default=0.4)
    parser.add_argument("--fitness_rank", default=False, action="store_true")

    parser.add_argument('--savetag', type=str, help='#Tag to append to savefile',  default='')
    parser.add_argument('--load_path', type=str, help='#load model', default='')
    parser.add_argument('--gpu_id', type=int, help='#GPU ID ',  default=0)
    parser.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=2)
    parser.add_argument('--buffer', type=float, help='Buffer size in million',  default=1.0)
    parser.add_argument('--frameskip', type=int, help='Frameskip',  default=1)

    parser.add_argument('--batchsize', type=int, help='Seed',  default=512)

    parser.add_argument('--reward_scale', type=float, help='Reward Scaling Multiplier',  default=1.0)
    parser.add_argument('--learning_start', type=int, help='Frames to wait before learning starts',  default=5000)

    #ALGO SPECIFIC ARGS
    parser.add_argument('--popsize', type=int, help='#Policies in the population',  default=10)
    parser.add_argument('--rollsize', type=int, help='#Policies in rollout size',  default=5)
    parser.add_argument('--gradperstep', type=float, help='#Gradient step per env step',  default=1.0)
    parser.add_argument('--num_test', type=int, help='#Test envs to average on',  default=5)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="mujoco.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser


def get_sac(env_constructor,args):
    # env_constructor = SafeEnvConstructor(args.env_name, args.frameskip)

    args.state_shape = env_constructor.observation_space.shape or env_constructor.observation_space.n
    args.action_shape = env_constructor.action_space.shape or env_constructor.action_space.n
    args.max_action = env_constructor.action_space.high[0]
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env_constructor.action_space.low), np.max(env_constructor.action_space.high))
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # model
    net_a = Net(args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(
        net_a,
        args.action_shape,
        max_action=args.max_action,
        device=args.device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = -np.prod(env_constructor.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = rcpoSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        estimation_step=args.n_step,
        action_space=env_constructor.action_space,
    )

    actor_builder = actor_constructor(args.state_shape,args.action_shape,args.hidden_sizes,args.max_action,policy)
    return policy,actor_builder


if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    # parser = argparse.ArgumentParser()
    args = get_args()
    args = Parameters(args)
    #######################  COMMANDLINE - ARGUMENTS ######################


    #Figure out GPU to use [Default is 0]


    #######################  Construct ARGS Class to hold all parameters ######################
    # args = Parameters(parser)
    # os.environ['CUDA_VISIBLE_DEVICES']=",".join(args.gpu_list)
    #Set seeds
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    ################################## Find and Set MDP (environment constructor) ########################

    # env_constructor = EnvConstructor(args.env_name, args.frameskip)
    env_constructor = SafeEnvConstructor(args.env_name, args.frameskip)

    sac_policy,actor_builder = get_sac(env_constructor,args)
    #######################  Actor, Critic and ValueFunction Model Constructor ######################
    # model_constructor = ModelConstructor(env_constructor.state_dim, env_constructor.action_dim, args.hidden_size)


    ai = ERL_Trainer(args, sac_policy, env_constructor, actor_builder)
    # ai.device = "cuda:%d"%vars(parser.parse_args())['gpu_id']
    ai.train(args.total_steps)


