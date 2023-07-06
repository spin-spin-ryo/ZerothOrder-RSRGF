import itertools
import os
from environments import CONFIGPATH
import json
from test import run

problem = "Quadratic"
property = "convex"
dim = 1000
rank = 100


lrs = [1,1e-1,1e-2,1e-3]
solver_name = "proposed"
mus = [1e-8]
sample_sizes = [1,10,100]
reduced_dims = [1,10,100]
iterations = 10000
interval = 1000
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
            "reduced_dim":reduced_dim,
            "mu": mu,
            "sample_size" : sample_size,
            "step_schedule" : "constant"
        },
        "iterations":iterations,
        "interval":interval
    }

    config_name = f"config{count}.json"
    with open(os.path.join(CONFIGPATH,config_name),"w") as f:
        json.dump(config_json,f)
        f.close()
    
    run(config_name)
