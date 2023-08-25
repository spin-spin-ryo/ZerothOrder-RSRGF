import itertools
import os
from environments import CONFIGPATH
import json
from test import run

problem = "regularized Quadratic"
property = "convex"
dim = 1000
rank = 500


lrs = [1e-6,1e-7,1e-8]
solver_name = "RGF"
mus = [1e-8]
sample_sizes = [1]
reduced_dims = [1]
iterations = 50000
interval = 10000
trial_numbers = 1
count = 0

for lr, reduced_dim,mu,sample_size in itertools.product(lrs,reduced_dims,mus,sample_sizes):
    config_json = {
        "problem":problem,
        "properties" : 
        {
            "property":property,
            "dim" : dim,
            "rank" : rank
        }
        ,
        "solver":solver_name,
        "params":
        {
            "lr" : lr,
            "mu": mu,
            "sample_size" : sample_size,
            "step_schedule" : "constant"
        },
        "iterations":iterations,
        "interval":interval,
        "trial_numbers":trial_numbers
    }

    config_name = f"config{count}.json"
    with open(os.path.join(CONFIGPATH,config_name),"w") as f:
        json.dump(config_json,f,indent=4)
        f.close()
    
    count += 1
    
    run(config_name)
