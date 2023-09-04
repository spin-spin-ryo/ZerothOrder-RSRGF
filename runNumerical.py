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
ord = 1
coef = None


# lrs = [1e-8,1e-9,1e-10,1e-11,1e-12]
lrs = [10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]

# solver_name = "RGF"
solver_name = "proposed"
mus = [1e-8]
sample_sizes = [10]
reduced_dims = [10,100]
iterations = 10000
interval = 100000
trial_numbers = 1
count = 0
data_num = None
fused_flag = False
step_schedule = "constant"



for lr, reduced_dim,mu,sample_size in itertools.product(lrs,reduced_dims,mus,sample_sizes):
    
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
            "step_schedule":step_schedule
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
