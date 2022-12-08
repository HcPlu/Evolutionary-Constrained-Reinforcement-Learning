
from core import utils as utils
import numpy as np
import torch
from torch import nn
from algos.simpleSAC import simpleSACPolicy
from torch.distributions import Independent, Normal
from tianshou.data import Batch
# device_list = ["cuda:4","cuda:5","cuda:6","cuda:7"]
device_num = torch.cuda.device_count()
device_list = ["cuda:%d"%i for i in range(device_num)]
# Rollout evaluate an agent in a complete game
@torch.no_grad()



def rollout_worker(id, type, task_pipe, result_pipe, store_data, model_bucket, env_constructor):

    env = env_constructor.make_env()
    env_name = env_constructor.env_name
    np.random.seed(id) ###make sure the random seeds across learners are different

    #RCPO
    # rcpo_lambda = 0.001

    ###LOOP###

    while True:
        recv_message = task_pipe.recv()  # Wait until a signal is received  to start rollout
        identifier = recv_message[0]
        rcpo_lambda = recv_message[1]
        if identifier == 'TERMINATE': exit(0) #Exit
        device = device_list[identifier % device_num]
        # print(device)
        # Get the requisite network
        net = model_bucket[identifier].to(device)
        policy = simpleSACPolicy(net)
        net.device = device
        reward_fitness = 0.0
        total_frame = 0
        state = env.reset()
        rollout_trajectory = []
        state = utils.to_tensor(state)
        constraint_fitness = 0
        while True:  # unless done

            input = Batch(obs=state, info={})
            action = policy(input)
            action = action.act
            action = utils.to_numpy(action.cpu())
            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment

            next_state = utils.to_tensor(next_state)
            reward_fitness += reward
            constraint = info["constraint"]
            constraint_fitness+=constraint

            if store_data: #Skip for test set
                saved_info = {"c":constraint,"rcpo_lambda":rcpo_lambda}
                rollout_trajectory.append(Batch(obs = utils.to_numpy(state),act = np.float32(action),rew = reward
                                                ,done = done,obs_next = utils.to_numpy(next_state),info =saved_info ))
            state = next_state
            total_frame += 1

            # DONE FLAG IS Received
            if done:
                break

        # Send back id, reward_fitness, constraint_fitness, total length and shaped reward_fitness using the result pipe
        if env_name == "safe-HalfCheetah-v1":
            constraint_fitness  = constraint_fitness
        else:
            constraint_fitness /= total_frame
        result_pipe.send([identifier, reward_fitness,constraint_fitness, total_frame, rollout_trajectory])
