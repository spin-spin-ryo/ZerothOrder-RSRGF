import itertools
import os
from environments import CONFIGPATH
import json
from test import run

problem = "regularized Quadratic"
property = "convex"
dim = 1000
rank = 500


lrs = [1e-6]
solver_name = "RGF"
# solver_name = "proposed"
mus = [1e-8,1e-5,1e-6,1e-7]
sample_sizes = [1]
reduced_dims = [100]
iterations = 50000
interval = 10000
trial_numbers = 100
count = 0
data_num = None
fused_flag = False

for lr, reduced_dim,mu,sample_size in itertools.product(lrs,reduced_dims,mus,sample_sizes):
    
    config_json = {
        "problem":"regularized Quadratic",
        "properties" : 
        {
            "dim" : dim,
            "data_num":data_num,
            "rank":rank,
            "property":property,
            "ord":1,
            "coef":1e-6,
            "fused": fused_flag
        }
        ,
        "solver":solver_name,
        "params":
        {
            "lr" : lr,
            "reduced_dim": reduced_dim,
            "sample_size":sample_size,
            "mu":mu,
            "step_schedule":"constant"
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
