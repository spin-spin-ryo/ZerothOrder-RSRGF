import itertools
import os
from environments import CONFIGPATH
import json
from main import run


problem = "subspace-norm local"
property = None
dim = 1000000
rank = None
subspace = 1000
bias = None
ord = 1
coef = None
data_name = None
data_num = None
fused_flag = None

lrs = [1e-8]
# lrs = [1e-3]

# solver_name = "RGF"
# solver_name = "proposed"
solver_name = "proposed-heuristic"
# solver_name = "AGD"

mus = [1e-8]
sample_sizes = [1]
reduced_dims = [100]
heuristic_intervals = [10,100]
iterations =200000
interval = 100000
trial_numbers = 1
count = 0
step_schedule = "constant"



for lr, reduced_dim,mu,sample_size,sample_interval in itertools.product(lrs,reduced_dims,mus,sample_sizes,heuristic_intervals):
    
    config_json = {
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
            "bias":bias
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
            "interval":sample_interval
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
