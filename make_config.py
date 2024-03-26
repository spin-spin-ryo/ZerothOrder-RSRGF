import itertools
import os
from environments import *
import json
from main import run
import subprocess
from utils import modifying_parameters

config_name = f"config.json"
  
problem = REGULARIZED+ROBUSTLOGISTIC
solver_name = RSRGF



property = None
dim = None
rank = None
subspace = None
bias = None
ord = 1
coef = 1e-7
fused_flag = False
data_name = "news20"
data_num = None
epoch_num = None
inner_iteration = 1000000
subproblem_eps = 1e-7
delta = 1e-3



lr = 1e-1
projection = True
reduced_dim = 10

mu = 1e-8
sample_size = 10
central = True


iterations =10000
interval = 10
trial_numbers = 1
count = 0
step_schedule = "constant"

if __name__ == "__main__":     
    config_json = {
        "problem":problem,
        "properties" : {},
        "solver":solver_name,
        "params":{}
    }
    all_config_json = {
      "problem":problem,
      "properties" : 
      {
          "dim" : dim,
          "data_num":data_num,
          "rank":rank,
          "property":property,
          "ord":ord,
          "subspace":subspace,
          "coef":coef,
          "fused": fused_flag,
          "data-name":data_name,
          "bias":bias,
          "epoch-num":epoch_num,
          "inner-iteration":inner_iteration,
          "subproblem-eps":subproblem_eps,
          "delta":delta
      }
      ,
      "solver":solver_name,
      "params":
      {
          "lr" : lr,
          "reduced_dim": reduced_dim,
          "sample_size":sample_size,
          "mu":mu,
          "step_schedule":step_schedule,
          "central":central,
          "projection":projection
      },
      "iterations":iterations,
      "interval":interval,
      "trial_numbers":trial_numbers
    }
    
    if REGULARIZED in problem:
        problem_without_regularized = problem.replace(REGULARIZED,"")
    else:
        problem_without_regularized = problem
    
    for key in OBJECTIVE_PARAMS_KEY[problem_without_regularized]:
        config_json["properties"][key] = all_config_json["properties"][key]
    
    if REGULARIZED in problem:
        for key in OBJECTIVE_PARAMS_KEY[REGULARIZED]:
            config_json["properties"][key] = all_config_json["properties"][key]
    
    for key in ALGORITHM_PARAMS_KEY[solver_name]:
        config_json["params"][key] = all_config_json["params"][key]
    
    
    with open(os.path.join(CONFIGPATH,config_name),"w") as f:
        json.dump(config_json,f,indent=4)
        f.close()
