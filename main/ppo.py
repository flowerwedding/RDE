"""
Created on  Nov 2 2023
@author: Zihang Zhang

"""
import array
import torch
import torch.nn as nn
# from seaborn.conftest import random_seed
from torch.distributions import Categorical, MultivariateNormal
import numpy as np
import random
import string
import matlab.engine

eng = matlab.engine.start_matlab()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("You are using:", device)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        # del语句作用在变量上，而不是数据对象上。删除的是变量，而不是数据。
        # del self.actions[:]
        # del self.states[:]
        # del self.logprobs[:]
        # del self.rewards[:]
        # del self.is_terminals[:]
        self.actions = [torch.tensor(action).to(device) for action in self.actions]
        self.states = [torch.tensor(state).to(device) for state in self.states]
        self.logprobs = [torch.tensor(logprob).to(device) for logprob in self.logprobs]
        self.rewards = [torch.tensor(reward).to(device) for reward in self.rewards]
        self.is_terminals = [torch.tensor(terminal).to(device) for terminal in self.is_terminals]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )
        # 方差
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        # 手动设置异常
        raise NotImplementedError

    def act(self, state, memory):
        with torch.no_grad():
            dist = Categorical(probs=self.actor(state))
            action = dist.sample()
            action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        # return action.numpy()[0], action_logprob.numpy()[0]
        return action.detach()

    def evaluate(self, state, action):
        state_value = self.critic(state)
        dist = Categorical(probs=self.actor(state))
        dist_entropy = dist.entropy()
        a_prob = self.actor(state)
        action_logprobs = np.argmax(a_prob.detach().cpu().numpy())
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        # return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        # old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        # old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class WindEnvironment:
    def __init__(self, i, offsPbest, popusize, state_dim, action_dim, dim, popuw, gbest, fitness_val, fitness_val_max,
                 fitness_old, r1, r2, rew):
        self.popusize = popusize
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rew = rew
        self.fitness_old = fitness_old
        self.dim = dim
        self.pbest = popuw
        self.gbest = gbest
        self.bestfit = fitness_val_max
        self.r1 = r1
        self.r2 = r2
        self.current_state = r1, r2 
        self.offsPbest = offsPbest
        self.fitness_val = fitness_val
        self.do = 0
        self.i = i
        self.gbest_current = gbest
        self.state_current = r1, r2

    def reset(self):
        self.current_state = np.zeros(self.state_dim)
        return self.current_state

    def step(self, action):
        if action == 0:
            self.r1 += 0.01
        elif action == 1:
            self.r1 -= 0.01
        elif action == 2:
            self.r2 += 0.01
        elif action == 3:
            self.r2 -= 0.01
        reward, gbest_current, state_current = self.calculate_reward()

        done = self.is_terminal()
        self.current_state = self.r1, self.r2
        return self.current_state, self.offsPbest, gbest_current, state_current, reward, done, {}

    def calculate_reward(self):
        if self.i > len(self.fitness_val):
            self.i = self.i % len(self.fitness_val)
            if self.i == 0:
                self.i = len(self.fitness_val)
        for d in range(self.dim):
            k = random.randint(1, self.popusize)
            k -= 1
            if self.fitness_val[self.i - 1] > self.fitness_val[k]:
                self.offsPbest[0, d] = (self.r2 * self.pbest[self.i - 1, d] +
                                        self.r1 * self.gbest[d])
            else:
                self.offsPbest[0, d] = self.pbest[k, d]
        fitness_val, population1, power_order, lp_power_accum = eng.wf_fitness_python(
            matlab.double(self.offsPbest.tolist()),
            nargout=4)

        population1 = np.array(population1)
        fitness_val_max = float(np.max(np.array(fitness_val)))
        if fitness_val_max > self.fitness_old:
            self.rew += 1.1
        elif fitness_val_max == self.fitness_old:
            self.rew = self.rew
        elif fitness_val_max < self.fitness_old:
            self.rew -= 1

        self.fitness_old = fitness_val_max
        if fitness_val_max > self.bestfit:
            # print('It works!!!', fitness_val_max, self.r1, self.r2) # TODO
            self.gbest_current = population1
            self.do = 2
            self.bestfit = fitness_val_max
            self.state_current = self.r1, self.r2

        return self.rew, self.gbest_current, self.state_current

    def is_terminal(self):
        return self.do > 1


def normalize(data, target_range):
    min_target, max_target = target_range
    min_data, max_data = np.min(data), np.max(data)
    normalized_data = (data - min_data) / (max_data - min_data) * (max_target - min_target) + min_target
    return normalized_data


def denormalize(normalized_data, original_range):
    min_orig, max_orig = original_range
    min_norm, max_norm = np.min(normalized_data), np.max(normalized_data)
    denormalized_data = (normalized_data - min_norm) / (max_norm - min_norm) * (max_orig - min_orig) + min_orig
    return denormalized_data


def main(tn, i, pbest, gbest, fitness_val, fitness_val_max, r1, r2, popusize, dim):
    ############## Hyperparameters ##############
    tn = int(tn)
    i = int(i)
    popusize = int(popusize)
    dim = int(dim)
    log_interval = 1  # print avg reward in the interval
    max_episodes = 100  # max training episodes
    max_timesteps = 100  # max timesteps in one episode

    update_timestep = 500  # update policy every n timesteps
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr = 0.001  # parameters for Adam optimizer
    betas = (0.9, 0.999)
    # random_seed = None
    #############################################
    # creating environment
    # instantiated Memory & PPO
    # dim is the number of the turbine
    state_dim = 2
    action_dim = 4
    # 创建一个3x4的初始列表
    fitness_old = 1.01
    offsPbest = np.zeros((1, tn))
    we = WindEnvironment(i, offsPbest, popusize, state_dim, action_dim, dim, pbest, gbest, fitness_val, fitness_val_max,
                         fitness_old, r1, r2, rew=-100)
    memory = Memory()
    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0
    # training loop
    done = False
    for i_episode in range(1, max_episodes + 1):
        # state = we.reset()
        # Reset
        state = r1, r2
        for t in range(max_timesteps):
            time_step += 1
            # Running policy_old:
            action = ppo.select_action(state, memory)
            state, offsPbest, gbest_current, state_current, reward, done, _ = we.step(action)
            # Saving reward and is_terminals:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if time_step % update_timestep == 0:
                ppo.update(memory)
                # memory.clear_memory()
                time_step = 0
            running_reward += reward
            # if render:
            #     we.render()
            # if done:
            #     break
        # if done:
        #     break

        avg_length += t
        env_name = 'wind'

        # save every 500 episodes
        if i_episode % 10 == 0:
            torch.save(ppo.policy.state_dict(), './PPO_GPSO_{}.pth'.format(env_name))

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            # print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

    return np.array(gbest_current), np.array(state_current)
