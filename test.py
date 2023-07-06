import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from environments import *
from generate_problem import generate
from get_solver import get_solver

config_name = "config.json"

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

func.SetDevice(DEVICE)
func.SetDtype(DTYPE)

x0 = x0.to(DTYPE).to(DEVICE)
problem_dir = ""
solver_dir = ""
for k,v in properties.items():
  problem_dir += k + ":" + v + "_"

for k,v in params_json.items():
  solver_dir += k + ":" + v + "_"

savepath = os.path.join(RESULTPATH,problem,problem_dir,solver_name,solver_dir)
os.makedirs(savepath,exist_ok= True)
x0.requires_grad_(True)


if __name__ == "__main__":
  solver.__iter__(func,x0,params,iterations,savepath,interval)
  plt.plot(solver.time_values,solver.fvalues)
  print("finished")
  plt.savefig(os.path.join(savepath,"result.png"))
  plt.savefig(os.path.join(savepath,"result.pdf"))