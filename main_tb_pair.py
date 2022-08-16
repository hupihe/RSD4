import numpy as np
import torch
import argparse
import utils
import random
import pandas as pd

from alg.RTD3_IN import RTD3_IN
from alg.RTD3_OUT import RTD3_OUT
from alg.RSD4_IN import RSD4_IN
from alg.RSD4_OUT import RSD4_OUT
from alg.RTD3_IN_XIN import RTD3_IN_XIN
from alg.RTD3_OUT_XIN import RTD3_OUT_XIN
from alg.RTD3_OUT_XIN_V1 import RTD3_OUT_XIN_V1
from alg.RTD3_OUT_XIN_CPU import RTD3_OUT_XIN_CPU
from alg.RSD4_IN_XIN import RSD4_IN_XIN
from alg.RSD4_OUT_XIN import RSD4_OUT_XIN
from alg.RSD4_OUT_XIN_V1 import RSD4_OUT_XIN_V1
from alg.RSD4_OUT_XIN_SB import RSD4_OUT_XIN_SB

from env.scheduler_simple import scheduler_simple
from env.scheduler_csi import scheduler_csi
from env.scheduler_mem import scheduler_mem
from env.scheduler_data import scheduler_data
from env.scheduler_datasimple import scheduler_datasimple


def get_full_action(policy, env, last_action, hidden_in):
    action = np.zeros(env.action_dim)
    full_action = np.zeros((env.tau_max + 1) * env.N)
    channel = env.channel_state
    hidden_out = []
    for i in range(env.N):
        st = np.sum(env.tau_n[:i], dtype='int') + i
        ed = np.sum(env.tau_n[:i + 1], dtype='int') + i + 1

        tmp = np.zeros(env.tau_max + 1)
        tmp[:ed - st] = env.Buffer[st:ed]
        if args.partial:
            mini_state = np.concatenate((np.array([i]), tmp))
        else:
            mini_state = np.concatenate((np.array([i]), tmp, channel[i:i + 1]))

        mini_last_action = np.zeros(env.tau_max + 1)
        mini_last_action[: ed - st] = last_action[st:ed]
        mini_action, mini_hidden_out = policy.select_action(mini_state, mini_last_action, hidden_in[i])
        action[st:ed] = mini_action[:env.tau_n[i] + 1]
        full_action[(env.tau_max + 1) * i: (env.tau_max + 1) * (i + 1)] = mini_action
        hidden_out.append(mini_hidden_out)
    return action, hidden_out

def get_full_action(policy, env, full_last_action, hidden_in):
    action = np.zeros(env.action_dim)
    full_action = np.zeros((env.tau_max + 1) * env.N)
    channel = env.channel_state
    hidden_out = []
    for i in range(env.N):
        st = np.sum(env.tau_n[:i], dtype='int') + i
        ed = np.sum(env.tau_n[:i + 1], dtype='int') + i + 1

        tmp = np.zeros(env.tau_max + 1)
        tmp[:ed - st] = env.Buffer[st:ed]
        if args.partial:
            mini_state = np.concatenate((np.array([i]), tmp))
        else:
            mini_state = np.concatenate((np.array([i]), tmp, channel[i:i + 1]))

        mini_last_action = full_last_action[(env.tau_max + 1) * i: (env.tau_max + 1) * (i+1)]
        mini_action, mini_hidden_out = policy.select_action(mini_state, mini_last_action, hidden_in[i])
        action[st:ed] = mini_action[:env.tau_n[i] + 1]
        full_action[(env.tau_max + 1) * i: (env.tau_max + 1) * (i+1)] = mini_action
        hidden_out.append(mini_hidden_out)
    return action, full_action, hidden_out

def eval_policy(policy, eval_episodes=1):
    if args.env == 'scheduler_simple':
        eval_env = scheduler_simple(lamb=args.lamb, user_num=args.user_num, time_step=args.train_len, seed1=args.seed1,
                                    seed2=args.seed2 + 100, partial=args.partial, homogeneous=args.homogeneous,
                                    period=args.period,
                                    cntype=args.cntype, square=args.square, poisson=args.poisson)
    elif args.env == 'scheduler_csi':
        eval_env = scheduler_csi(lamb=args.lamb, user_num=args.user_num, time_step=args.train_len, seed1=args.seed1,
                                 seed2=args.seed2 + 100, partial=args.partial, homogeneous=args.homogeneous,
                                 period=args.period,
                                 cntype=args.cntype, square=args.square, poisson=args.poisson)
    elif args.env == 'scheduler_mem':
        eval_env = scheduler_mem(lamb=args.lamb, user_num=args.user_num, time_step=args.train_len, seed1=args.seed1,
                                 seed2=args.seed2 + 100, partial=args.partial, homogeneous=args.homogeneous,
                                 period=args.period,
                                 cntype=args.cntype, square=args.square, poisson=args.poisson)
    elif args.env == 'scheduler_data':
        eval_env = scheduler_data(arrival_data_path, channel_data_path, lamb=args.lamb, user_num=args.user_num,
                                  time_step=args.train_len, seed1=args.seed1, seed2=args.seed2 + 100,
                                  partial=args.partial,
                                  homogeneous=args.homogeneous)
    elif args.env == 'scheduler_datasimple':
        eval_env = scheduler_datasimple(arrival_data_path, channel_data_path, lamb=args.lamb, user_num=args.user_num,
                                        time_step=args.train_len, seed1=args.seed1, seed2=args.seed2 + 100,
                                        partial=args.partial,
                                        homogeneous=args.homogeneous)
    eval_env.action_space.np_random.seed(args.seed2+100)
    all_episode_rewards = []
    all_episode_throughput = []
    all_episode_energy = []
    for episode_idx in range(eval_episodes):
        episode_rewards = []
        state = eval_env.reset()
        throughput = 0
        energy = 0
        done = False
        last_action = eval_env.action_space.sample()
        full_last_action = np.zeros((eval_env.tau_max + 1) * eval_env.N)
        for i in range(eval_env.N):
            st = np.sum(eval_env.tau_n[:i], dtype='int') + i
            ed = np.sum(eval_env.tau_n[:i + 1], dtype='int') + i + 1
            full_last_action[i * (eval_env.tau_max + 1): i * (eval_env.tau_max + 1) + eval_env.tau_n[i] + 1] = last_action[st: ed]
        if 'CPU' in args.policy:
            hidden_out = [(np.zeros([args.num_layers, 1, hidden_dim]), np.zeros([args.num_layers, 1, hidden_dim]))] * eval_env.N
        else:
            hidden_out = [(torch.zeros([args.num_layers, 1, hidden_dim], dtype=torch.float).to(device), torch.zeros([args.num_layers, 1, hidden_dim], dtype=torch.float).to(device))] * eval_env.N
        while not done:
            hidden_in = hidden_out
            action, full_action, hidden_out = get_full_action(policy, eval_env, full_last_action, hidden_in)
            full_last_action = full_action
            state, reward, done, info = eval_env.step(action)
            episode_rewards.append(reward)
            throughput = info['throughput']
            energy = info['energy']

        all_episode_rewards.append(np.mean(episode_rewards))
        all_episode_throughput.append(throughput)
        all_episode_energy.append(energy)

    full_last_action = np.zeros((eval_env.tau_max + 1) * eval_env.N) + min_action
    if 'CPU' in args.policy:
        hidden_in = [(np.zeros([args.num_layers, 1, hidden_dim]), np.zeros([args.num_layers, 1, hidden_dim]))] * eval_env.N
    else:
        hidden_in = [(torch.zeros([args.num_layers, 1, hidden_dim], dtype=torch.float).to(device), torch.zeros([args.num_layers, 1, hidden_dim], dtype=torch.float).to(device))] * eval_env.N

    print('Evaluated Action')
    for i in range(env.N):
        mini_last_action = full_last_action[(eval_env.tau_max + 1) * i: (eval_env.tau_max + 1) * (i+1)]
        if args.partial:
            action, hidden_out = policy.select_action(np.concatenate((np.array([i]), np.ones(eval_env.tau_max + 1))), mini_last_action, hidden_in[i])
        else:
            action, hidden_out = policy.select_action(np.concatenate((np.array([i]), np.ones(eval_env.tau_max + 1), np.array([0]))), mini_last_action, hidden_in[i])
        print(i, action[:env.tau_n[i] + 1])

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    mean_episode_throughput = np.mean(all_episode_throughput)
    std_episode_throughput = np.std(all_episode_throughput)
    mean_episode_energy = np.mean(all_episode_energy)
    std_episode_energy = np.std(all_episode_energy)

    del hidden_out
    del hidden_in
    # return mean_episode_reward, std_episode_reward, mean_episode_throughput, std_episode_throughput, mean_episode_energy, std_episode_energy
    return mean_episode_reward, mean_episode_throughput, mean_episode_energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="tmp_test")
    parser.add_argument("--policy", default="RSD4")
    parser.add_argument("--lamb", default=1, type=float)
    parser.add_argument("--env", default="scheduler")
    parser.add_argument('--partial', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--homogeneous', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--poisson', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--period", default=0, type=int)
    parser.add_argument("--square", default=0, type=int)
    parser.add_argument("--cntype", default=0, type=int)
    parser.add_argument("--user-num", default=4, type=int)
    parser.add_argument("--limit", default=10, type=float)
    parser.add_argument("--seed1", default=1, type=int)
    parser.add_argument("--seed2", default=1, type=int)
    parser.add_argument("--start-steps", default=10000, type=int, help='Number of steps for the warm-up stage using random policy')
    parser.add_argument("--train-freq", default=100, type=int, help='Update the model every train-freq steps')
    parser.add_argument("--eval-freq", default=100, type=int, help='Number of steps per evaluation')
    parser.add_argument("--update-itr", default=20, type=int)
    parser.add_argument("--train-len", default=100, type=int)
    parser.add_argument("--eval-len", default=1000, type=int)
    parser.add_argument("--episode-num", default=100000, type=int, help='Maximum number of episodes')
    parser.add_argument("--discount", default=1, type=float, help='Discount factor')
    parser.add_argument("--tau", default=0.05, type=float)
    parser.add_argument("--actor-lr", default=3e-3, type=float)
    parser.add_argument("--critic-lr", default=3e-3, type=float)
    parser.add_argument("--num-layers", default=1, type=int)
    parser.add_argument("--hidden-dim", default=32, type=int)
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument('--save-model', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--load-model", default="")
    parser.add_argument("--action-noise", default=0.1, type=float)
    parser.add_argument("--policy-noise", default=0.2, type=float)
    parser.add_argument("--weight-decay", default=0.1, type=float)
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--dropout-nn", default=0, type=float)
    parser.add_argument("--policy-freq", default=2, type=int, help='Frequency of delayed policy updates')
    parser.add_argument('--beta', default='.01', type=float, help='The parameter beta in softmax')
    parser.add_argument('--num-noise-samples', type=int, default=50, help='The number of noises to sample for each next_action')
    parser.add_argument('--imps', type=int, default=0, help='Whether to use importance sampling for gaussian noise when calculating softmax values')
    parser.add_argument('--device-idx', default=0, type=int)
    parser.add_argument('--arrival', default="LTE_dataset_low")
    parser.add_argument('--channel', default="channel_state")
    args = parser.parse_args()

    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed1: {}, Seed1: {}".format(args.policy, args.env, args.seed1, args.seed2))
    print("---------------------------------------")

    model_path = 'tmp_model/' + args.name
    data_path = 'tmp_data/' + args.name
    arrival_data_path = 'dataset_dir/' + args.arrival + '.csv'
    channel_data_path = 'dataset_dir/' + args.channel + '.csv'

    if args.env == 'scheduler_simple':
        env = scheduler_simple(lamb=args.lamb, user_num=args.user_num, time_step=args.train_len, seed1=args.seed1,
                               seed2=args.seed2, partial=args.partial, homogeneous=args.homogeneous, period=args.period,
                               cntype=args.cntype, square=args.square, poisson=args.poisson)
    elif args.env == 'scheduler_csi':
        env = scheduler_csi(lamb=args.lamb, user_num=args.user_num, time_step=args.train_len, seed1=args.seed1,
                            seed2=args.seed2, partial=args.partial, homogeneous=args.homogeneous, period=args.period,
                            cntype=args.cntype, square=args.square, poisson=args.poisson)
    elif args.env == 'scheduler_mem':
        env = scheduler_mem(lamb=args.lamb, user_num=args.user_num, time_step=args.train_len, seed1=args.seed1,
                            seed2=args.seed2, partial=args.partial, homogeneous=args.homogeneous, period=args.period,
                            cntype=args.cntype, square=args.square, poisson=args.poisson)
    elif args.env == 'scheduler_data':
        env = scheduler_data(arrival_data_path, channel_data_path, lamb=args.lamb, user_num=args.user_num,
                             time_step=args.train_len, seed1=args.seed1, seed2=args.seed2, partial=args.partial,
                             homogeneous=args.homogeneous)
    elif args.env == 'scheduler_datasimple':
        env = scheduler_datasimple(arrival_data_path, channel_data_path, lamb=args.lamb, user_num=args.user_num,
                                   time_step=args.train_len, seed1=args.seed1, seed2=args.seed2, partial=args.partial,
                                   homogeneous=args.homogeneous)
    torch.manual_seed(args.seed2)
    np.random.seed(args.seed2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed2)
    random.seed(args.seed2)
    env.action_space.np_random.seed(args.seed2)

    state_dim = env.tau_max + 2 if args.partial else env.tau_max + 3
    action_dim = env.tau_max + 1
    max_action = float(env.action_space.high[0])
    min_action = -max_action
    hidden_dim = args.hidden_dim

    if args.device_idx == -1:
        GPU = False
    else:
        GPU = True
    device_idx = args.device_idx
    if GPU:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    torch.cuda.set_device(device_idx)

    print(device)

    #if args.user_num > 20:
    #    torch.set_num_threads(1)

    replay_buffer_size = 1e4

    if args.update_itr == 0:
        args.update_itr = 1
    else:
        args.update_itr = env.N
    print('ui', args.update_itr)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_dim": args.hidden_dim,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "device": device,
    }

    if args.policy == "RSD4_IN":
        kwargs['beta'] = args.beta
        kwargs['with_importance_sampling'] = args.imps
        kwargs["policy_noise"] = args.policy_noise
        kwargs['num_noise_samples'] = args.num_noise_samples
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RSD4_IN(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RSD4_OUT":
        kwargs['beta'] = args.beta
        kwargs['with_importance_sampling'] = args.imps
        kwargs["policy_noise"] = args.policy_noise
        kwargs['num_noise_samples'] = args.num_noise_samples
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RSD4_OUT(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RTD3_IN":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RTD3_IN(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RTD3_OUT":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RTD3_OUT(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RSD4_IN_XIN":
        kwargs['beta'] = args.beta
        kwargs['with_importance_sampling'] = args.imps
        kwargs["policy_noise"] = args.policy_noise
        kwargs['num_noise_samples'] = args.num_noise_samples
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RSD4_IN_XIN(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RSD4_OUT_XIN":
        kwargs['beta'] = args.beta
        kwargs['with_importance_sampling'] = args.imps
        kwargs["policy_noise"] = args.policy_noise
        kwargs['num_noise_samples'] = args.num_noise_samples
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RSD4_OUT_XIN(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RSD4_OUT_XIN_V1":
        kwargs['beta'] = args.beta
        kwargs['with_importance_sampling'] = args.imps
        kwargs["policy_noise"] = args.policy_noise
        kwargs['num_noise_samples'] = args.num_noise_samples
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RSD4_OUT_XIN_V1(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RSD4_OUT_XIN_SB":
        kwargs['beta'] = args.beta
        kwargs['with_importance_sampling'] = args.imps
        kwargs["policy_noise"] = args.policy_noise
        kwargs['num_noise_samples'] = args.num_noise_samples
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RSD4_OUT_XIN_SB(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RTD3_IN_XIN":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RTD3_IN_XIN(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RTD3_OUT_XIN":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RTD3_OUT_XIN(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RTD3_OUT_XIN_V1":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RTD3_OUT_XIN_V1(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB(replay_buffer_size)
    elif args.policy == "RTD3_OUT_XIN_CPU":
        kwargs["policy_noise"] = args.policy_noise
        kwargs["policy_freq"] = args.policy_freq
        kwargs["num_layers"] = args.num_layers
        kwargs["weight_decay"] = args.weight_decay
        kwargs["dropout"] = args.dropout
        policy = RTD3_OUT_XIN_CPU(**kwargs)
        replay_buffer = utils.ReplayBufferLSTM_TB_CPU(replay_buffer_size)
    if args.load_model != "":
        policy.load("./models/{}".format(args.load_model))

    best_reward = -np.inf
    best_throughput = 0
    best_energy = 0

    df = pd.DataFrame(columns=['eps', 'reward', 'throughput', 'energy', 'avg_reward', 'avg_throughput', 'avg_energy'])
    frame_idx = 0

    for eps in range(int(args.episode_num)):
        state = env.reset()
        last_action = env.action_space.sample()
        full_last_action = np.zeros((env.tau_max + 1) * env.N)
        for i in range(env.N):
            st = np.sum(env.tau_n[:i], dtype='int') + i
            ed = np.sum(env.tau_n[:i + 1], dtype='int') + i + 1
            full_last_action[i * (env.tau_max + 1): i * (env.tau_max + 1) + env.tau_n[i] + 1] = last_action[st: ed]
        episode_state = [[] for _ in range(env.N)]
        episode_action = [[] for _ in range(env.N)]
        episode_last_action = [[] for _ in range(env.N)]
        episode_reward = [[] for _ in range(env.N)]
        episode_next_state = [[] for _ in range(env.N)]
        episode_done = [[] for _ in range(env.N)]
        if 'CPU' in args.policy:
            hidden_out = [(np.zeros([args.num_layers, 1, hidden_dim]), np.zeros([args.num_layers, 1, hidden_dim]))] * env.N
        else:
            hidden_out = [(torch.zeros([args.num_layers, 1, hidden_dim], dtype=torch.float).to(device), torch.zeros([args.num_layers, 1, hidden_dim], dtype=torch.float).to(device))] * env.N

        for step in range(int(args.train_len)):
            hidden_in = hidden_out
            if frame_idx < args.start_steps:
                action = (max_action - min_action) * np.random.random(env.action_space.shape) + min_action
                full_action = np.zeros((env.tau_max + 1) * env.N)
                for i in range(env.N):
                    st = np.sum(env.tau_n[:i], dtype='int') + i
                    ed = np.sum(env.tau_n[:i + 1], dtype='int') + i + 1
                    full_action[i * (env.tau_max + 1): i * (env.tau_max + 1) + env.tau_n[i] + 1] = action[st: ed]
            else:
                action, full_action, hidden_out = get_full_action(policy, env, full_last_action, hidden_in)
                action = (action + np.random.normal(0, max_action * args.action_noise, size=env.action_dim)).clip(-max_action, max_action)
            next_state, reward, done, info = env.step(action)

            #print('after', step, state, next_state, env.Buffer)
            if step == 0:
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out

            channel = state[env.action_dim:]
            next_channel = next_state[env.action_dim:]
            for i in range(env.N):
                st = np.sum(env.tau_n[:i], dtype='int') + i
                ed = np.sum(env.tau_n[:i + 1], dtype='int') + i + 1
                tmp = np.zeros(env.tau_max + 1)
                tmp[:ed - st] = state[st:ed]
                mini_state = np.concatenate((np.array([i]), tmp, channel[i:i + 1]))
                mini_action = full_action[(env.tau_max + 1) * i: (env.tau_max + 1) * (i+1)]
                mini_last_action = full_last_action[(env.tau_max + 1) * i: (env.tau_max + 1) * (i+1)]
                mini_reward = info['reward'][i]
                next_tmp = np.zeros(env.tau_max + 1)
                next_tmp[:ed - st] = next_state[st:ed]
                mini_next_state = np.concatenate((np.array([i]), next_tmp, next_channel[i:i + 1]))
                mini_done = done

                #print('state: ', mini_state, 'action: ', mini_action, 'last_action: ', mini_last_action, 'reward: ', mini_reward, 'next_state: ', mini_next_state, 'done: ', mini_done)
                episode_state[i].append(mini_state)
                episode_action[i].append(mini_action)
                episode_last_action[i].append(mini_last_action)
                episode_reward[i].append(mini_reward)
                episode_next_state[i].append(mini_next_state)
                episode_done[i].append(mini_done)

            state = next_state
            last_action = action
            full_last_action = full_action
            frame_idx += 1

            if len(replay_buffer) > args.batch_size and frame_idx % args.train_freq == 0 and eps > 0 and frame_idx >= args.start_steps:
                for i in range(args.update_itr):
                    policy.train(replay_buffer, args.batch_size)

        for i in range(env.N):
            replay_buffer.push(ini_hidden_in[i], ini_hidden_out[i], episode_state[i], episode_action[i], episode_last_action[i], episode_reward[i], episode_next_state[i], episode_done[i])

        mean_reward = np.sum(np.mean(episode_reward, axis=1))

        if eps % args.eval_freq == 0:
            df.to_csv(data_path + '.csv', index=False)
            avg_reward, avg_throughput, avg_energy = eval_policy(policy)
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_throughput = avg_throughput
                best_energy = avg_energy
                if args.save_model:
                    print('Saving model at eps:', eps, 'with reward: {:.2f}'.format(avg_reward))
                    policy.save(model_path)
            print(args.name, 'Eval_Episode: ', eps, "| Eval_reward: {:.2f} - Eval_throughput: {:.2f} - Eval_energy: {:.2f}".format(avg_reward, avg_throughput, avg_energy))

        df.loc[eps] = [eps, mean_reward, info['throughput'], info['energy'], avg_reward, avg_throughput, avg_energy]
        print(args.name, 'Episode: ', eps, "| Best reward: {:.2f} - Current reward: {:.2f} - Throughput: {:.2f} - Energy: {:.2f}".format(best_reward, mean_reward, info['throughput'], info['energy']))
        #torch.cuda.empty_cache()