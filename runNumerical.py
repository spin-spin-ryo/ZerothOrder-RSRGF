import itertools
import os
from environments import CONFIGPATH
import json
from main import run


problem = "regularized softmax"
property = None
dim = None
rank = None
subspace = None
bias = None
ord = 1
coef = 1e-5
data_name = "Scotus"
data_num = None
fused_flag = False

# lrs = [1000,100,10,1,1e-1,1e-2,1e-3,1e-4,1e-5]
# lrs = [1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
# lrs = [1e-11,1e-12,1e-13,1e-14]
lrs = [100]

# solver_name = "RGF"
solver_name = "proposed"
# solver_name = "proposed-heuristic"
# solver_name = "AGD"

mus = [1e-12]
sample_sizes = [1]
reduced_dims = [10]
heuristic_intervals = [None]
iterations =1000000
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
