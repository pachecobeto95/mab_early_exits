import random
import numpy as np
import pandas as pd

def reward(early_exit_conf, threshold, delta_conf, overhead):
  if(early_exit_conf >= threshold):
    reward = 0
  else:
    reward = delta_conf - overhead
  return reward

class Regret(object):
  def __init__(self):
    self.regret = 0
    self.acc_regret = 0
  
  def update(self, delta_conf, overhead, reward):
    optimal_reward = max(0, delta_conf - overhead)
    
    self.regret = optimal_reward - reward
    self.acc_regret += self.regret



class Bandit(object):
  # Base Badit Class

  def __init__(self, total_arms):
    self.total_arms   = total_arms
    self.arms_rewards =  {}
        
  def act(self):
    pass
    
  def update(self, arm, reward):
    if arm in self.arms_rewards:
      self.arms_rewards[arm].append(reward)
    else:
      self.arms_rewards[arm] = [reward]
    
  def compute_average_rewards(self):
    avg_reward = []
    for key in self.arms_rewards.keys():
      reward_history = self.arms_rewards[key]
      avg_reward.append(np.mean(reward_history))

    return np.array(avg_reward)

class RandomPolicy(Bandit):
  # Random Select ARM
  #
  def __init__(self, total_arms, seed=42):
    super().__init__(total_arms, seed=42)

  def act(self):
    return random.choice(list(self.arms_rewards.keys()))


class EpsilonGreedy(Bandit):
  def __init__(self, total_arms, epsilon=0.1):
    super().__init__(total_arms)
    self.total_arms = total_arms
    self.epsilon = epsilon

  def act(self):

    if (random.choice([True, False], p=[self.epsilon, 1.0 - self.epsilon])):
      action = random.randint(0, self.total_arms)
    else:
      action = np.argmax(self.reduction_rewards())

    return action


class UCB(Bandit):
  def __init__(self, arms, total_arms, c):
    super(UCB, self).__init__(total_arms)

    self.c = c
    self.arms = arms
    self.total_arms = total_arms
    self.t = 1
    self.action_times = np.zeros(total_arms)
  
  def act(self):
    reward_mean = self.compute_average_rewards()
    confidence_bound = reward_mean + self.c * np.sqrt(np.log(self.t)/(self.action_times + 0.001))
    action = np.argmax(confidence_bound)

    self.t += 1
    self.action_times[action] += 1

    return self.arms[action]

def run_mab(df, arms, func_reward, agent, label, overhead, verbose):
  if (label != "all"):
    df = df[df.label == label]
  
  df = df.sample(frac=1)

  arms_rewards  = {}
  rewards, arms_selected, regret_list = [], [], []

  regret = Regret()

  # init rewards
  for arm in arms:
    agent.update(arm, 0)
  
  for step, (_, row) in enumerate(df.iterrows(), 1):
    arm = agent.act()
    reward = func_reward(row.conf_branch_1, arm, row.delta_conf, overhead)

    agent.update(arm, reward)    
    regret.update(row.delta_conf, overhead, reward)
    
    arms_selected.append(arm)
    rewards.append(reward)
    regret_list.append(regret.regret)
 
    if (verbose):
      print("Step: %s, Arm: %s, Reward: %s"%(step, arm, reward))

    
  if (verbose):
    print("Reward Cum: %s"%(np.sum(rewards)))
    
  result = {"arm_selected": arms_selected, "rewards": rewards, "regret": regret_list, 
            "label": [label]*len(rewards), "overhead": [overhead]*len(rewards)}
  
  return result
