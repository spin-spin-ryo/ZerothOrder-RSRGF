import itertools
import os
from environments import CONFIGPATH
import json
from main import run,make_shfile
import subprocess

split_sh = True

problem = "regularized robust adversarial"
property = None
dim = None
rank = None
subspace = None
bias = None
ord = 1
coef = 1e-6
data_name = "news20"
data_num = None
fused_flag = False
epoch_num = None
inner_iteration = 100000
subproblem_eps = 1e-3

# lrs = [1e-3]
lrs = [200,300,400]

# solver_name = "RGF"
# solver_name = "OZD"
solver_name = "proposed"
# solver_name = "proposed-heuristic"
# solver_name = "proposed-sparse"
# solver_name = "AGD"

mus = [1e-8]
sample_sizes = [1]
reduced_dims = [10]
# reduced_dims = [None]
heuristic_intervals = [None]
sparsity = None
central = None

iterations =10000
interval = 1000
trial_numbers = 1
count = 0
step_schedule = "constant"


if __name__ == "__main__":

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
                "bias":bias,
                "epoch-num":epoch_num,
                "inner-iteration":inner_iteration,
                "subproblem-eps":subproblem_eps
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
                "interval":sample_interval,
                "sparsity":sparsity,
                "central":central
            },
            "iterations":iterations,
            "interval":interval,
            "trial_numbers":trial_numbers
        }

        if not split_sh:
            config_name = f"config{count}.json"
            with open(os.path.join(CONFIGPATH,config_name),"w") as f:
                json.dump(config_json,f,indent=4)
                f.close()
            count += 1
            run(config_name)
        else:
            count = len(os.listdir(os.path.join(CONFIGPATH,"auto"))) + len(os.listdir(os.path.join(CONFIGPATH,"done")))
            config_name = f"config{count}.json"
            with open(os.path.join(CONFIGPATH,"auto",config_name),"w") as f:
                json.dump(config_json,f,indent=4)
                f.close()
            sh_file_name = f"run{count}.sh"
            make_shfile(sh_file_name,config = config_name)
            subprocess.run(["sbatch",f"./shfiles/{sh_file_name}"])
            