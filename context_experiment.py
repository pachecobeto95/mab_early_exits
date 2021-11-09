import pandas as pd
import numpy as np
import itertools, os
from multi_armed_bandits import run_mab, UCB, reward

def exp_early_exit_context_mab(df, arms, reward, agent, n_rounds, label_list, overhead_list, savePath, verbose=False):
  
  df_result = pd.DataFrame()
  
  for n in range(n_rounds):
    print("Round: %s"%(n))
    config_list = list(itertools.product(*[label_list, overhead_list]))
    
    for i, (label, overhead) in enumerate(config_list):
      print("N Configuration: %s" %(i))
      result = run_mab(df, arms, reward, agent, label, overhead, verbose)
      df2 = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
      df_result = df_result.append(df2)

      df_result.to_csv(savePath)

def evaluating_ucb_parameters(df, arms, reward, experimentsDict, n_rounds, label_list, overhead_list, verbose=False):
  for name, ucb in experimentsDict.items():
    exp_early_exit_context_mab(df, arms, reward, ucb["agent"], n_rounds, label_list, overhead_list, ucb["save_path"], verbose=verbose)


dataset_name = "cifar_10"
model_name = "mobilenet"
results_path = os.path.join(".", "results")

n_classes = 10
threshold_list = [0.7, 0.75, 0.8, 0.85, 0.9]
total_arms = len(threshold_list)
overhead_list = np.arange(0, 0.5, 0.05)
n_rounds = 10
verbose = False

data_path = os.path.join(".", "inference_experiment_get_data_ucb_3_pre.csv")

df_result = pd.read_csv(data_path)
df_result = df_result.loc[:, ~df_result.columns.str.contains('^Unnamed')]

label_list = ["cat", "dog", "ship", "automobile", "all"]


experimentsDict= {"UCB_0.1": {"agent": UCB(threshold_list, total_arms, c=0.1), "save_path": os.path.join(results_path, "results_ucb_0_1.csv")},
                  "UCB_0.5": {"agent": UCB(threshold_list, total_arms, c=0.5), "save_path": os.path.join(results_path, "results_ucb_0_5.csv")},
                  "UCB_1.0": {"agent": UCB(threshold_list, total_arms, c=1.0), "save_path": os.path.join(results_path, "results_ucb_1_0.csv")},
                  "UCB_1.5": {"agent": UCB(threshold_list, total_arms, c=1.5), "save_path": os.path.join(results_path, "results_ucb_1_5.csv")},
                  "UCB_2.0": {"agent": UCB(threshold_list, total_arms, c=2.0), "save_path": os.path.join(results_path, "results_ucb_2_0.csv")}}

evaluating_ucb_parameters(df_result, threshold_list, reward, experimentsDict, n_rounds, label_list, overhead_list, verbose=verbose)