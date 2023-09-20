import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from environments import *
from generate_problem import generate
from get_solver import get_solver
import sys
import logging
from summarizing.modify_json import get_element_json,save_result_json,count_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



def run(config_name,save_solution = False):
  if not os.path.exists(os.path.join(CONFIGPATH,config_name)):
    raise ValueError("No config")

  with open(os.path.join(CONFIGPATH,config_name)) as f:
    config = json.load(f)

  problem = config["problem"]
  properties = config["properties"]
  solver_name = config["solver"]
  params_json = config["params"]


  func,x0 = generate(mode = problem,properties = properties)

  solver,params = get_solver(solver_name=solver_name,params_json=params_json)
  solver.device = DEVICE
  solver.dtype = DTYPE

  iterations = int(config["iterations"])
  interval = int(config["interval"])
  trial_numbers = config["trial_numbers"]


  func.SetDevice(DEVICE)
  func.SetDtype(DTYPE)

  x0 = x0.to(DTYPE).to(DEVICE)
  problem_dir = ""
  solver_dir = ""
  for k,v in properties.items():
    if v is not None:
      problem_dir += k + ":" + str(v) + "_"
  problem_dir = problem_dir[:-1]

  for k,v in params_json.items():
    if v is not None:
      solver_dir += k + ":" + str(v) + "_"
  solver_dir = solver_dir[:-1]
  savepath = os.path.join(RESULTPATH,problem,problem_dir,solver_name,solver_dir)
  logger.info(savepath)
  os.makedirs(savepath,exist_ok= True)
  for i in range(trial_numbers):
    x = x0.clone().detach()
    x.requires_grad_(True)
    solver.__iter__(func,x,params,iterations,savepath,interval)
    logger.info(f"{iterations}")
    values_dict = get_element_json(solver.save_values)
    for k,v in values_dict.items():
      logger.info(f"{k}:{v}")
    save_result_json(os.path.join(savepath,"result.json"),values_dict,iterations)
    count = count_files(savepath,r"fvalues.*\.pth")
    if count == 0:
      suffix = ""
    else:
      suffix = str(count)
    for k,v in solver.save_values.items():
      torch.save(v,os.path.join(savepath,k[0]+suffix+".pth"))
  
  # 最後の反復の結果だけ保存
  if save_solution:
    torch.save(solver.xk,os.path.join(savepath,"solution.pth"))
  fvalues = None
  timevalues = None
  count = count_files(savepath,r"fvalues.*\.pth")
  if count == 0:
    suffix = ""
  else:
    suffix = str(count)
  for k,v in solver.save_values.items():
    if k[0] == "fvalues":
      fvalues = v
    if k[0] == "time_values":
      timevalues = v
    if k[0] == "norm_dir":
      plt.plot(np.arange(len(v)),v)
      plt.yscale("log")
      plt.savefig(os.path.join(savepath,"norm_dir.png"))
      plt.close()
  plt.plot(timevalues,fvalues)
  plt.savefig(os.path.join(savepath,"result.png"))
  plt.savefig(os.path.join(savepath,"result.pdf"))
  plt.close()
  
def make_shfile(sh_file_name,**kwargs):
  try:
    config_name = kwargs["config"]
  except:
    config_name = "config.json"
  
  with open(os.path.join(SHPATH,sh_file_name),"w") as f:
    f.write("#!/bin/sh\n")
    f.write("#SBATCH -p v\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH -N 1\n")
    f.write(f"#SBATCH -o /home/u00786/Research/optimization/outputs/{config_name}.out\n")
    f.write("export PATH=/home/app/singularity-ce/bin:$PATH\n")
    f.write(f"singularity exec --nv /home/u00786/Research/Master-Research1/pytorch_22.03-py3.sif python /home/u00786/Research/optimization/main.py {os.path.join('auto',config_name)}\n")
    f.write(f"mv {os.path.join('/home/u00786/Research/optimization',CONFIGPATH,'auto',config_name)} {os.path.join('/home/u00786/Research/optimization',CONFIGPATH,'done',config_name)}")
    f.close()


if __name__ == "__main__":
  import sys
  args = sys.argv
  config_name = args[1]
  run(config_name)