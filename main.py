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
  result_json = {"result":[]}
  for i in range(trial_numbers):
    x = x0.clone().detach()
    x.requires_grad_(True)
    solver.__iter__(func,x,params,iterations,savepath,interval)
    result_dict = {}
    logger.info(f"{iterations}")
    for k,v in solver.save_values.items():
      if k[1] == "min":
        result_dict[k[0]] = torch.min(v).item()
      elif k[1] == "max":
        result_dict[k[0]] = torch.max(v).item()
    for k,v in result_dict.items():
      logger.info(f"{k}:{v}")
    result_json["result"].append(result_dict)


  # 最後の反復の結果だけ保存
  if save_solution:
    torch.save(solver.xk,os.path.join(savepath,"solution.pth"))
  fvalues = None
  timevalues = None
  for k,v in solver.save_values.items():
    torch.save(v,os.path.join(savepath,k[0]+".pth"))
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
  min_values = []
  for each_result in result_json['result']:
    for k,v in each_result.items():
      if k == "fvalues":
        min_values.append(v)
  
  min_values = np.array(min_values)
  result_json["mean"] = min_values.mean()
  result_json["std"] = min_values.std()
  with open(os.path.join(savepath,"result.json"),"w") as f:
    json.dump(result_json,f,indent=4)
    f.close()
  
  



if __name__ == "__main__":
  config_name = "config.json"
  run(config_name,save_solution=True)