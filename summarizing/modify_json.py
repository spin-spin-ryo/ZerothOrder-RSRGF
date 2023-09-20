import json
import os
import torch
import numpy as np
import re

def change_json_attribute(json_path,**kwargs):
    with open(json_path,"r") as f:
        json_dict = json.load(f)
    print(json_dict)
    new_json_dict = {}
    new_json_dict["result"] = []
    element_json_dict = {}
    if "iteration" in kwargs.keys():
        element_json_dict["iteration"] = kwargs["iteration"]
    element_json_dict["save_values"] = json_dict["result"]
    element_json_dict["mean"] = json_dict["mean"]
    element_json_dict["std"] = json_dict["std"]
    new_json_dict["result"].append(element_json_dict)
    with open(json_path,"w") as f2:
        json.dump(new_json_dict,f2,indent=4)

def change_json_in_dir(init_dir):
    for curDir, dirs, files in os.walk(init_dir):
        for file_name in files:
            if file_name == "result.json":
                json_path = os.path.join(curDir,file_name)
                fvalues_path = os.path.join(curDir,"fvalues.pth")
                fvalues = torch.load(fvalues_path)
                iteration = len(fvalues)
                change_json_attribute(json_path,iteration = iteration)

def get_element_json(save_values):
    values_dict = {}
    for k,v in save_values.items():
      if k[1] == "min":
        values_dict[k[0]] = torch.min(v).item()
      elif k[1] == "max":
        values_dict[k[0]] = torch.max(v).item()
    return values_dict

def save_result_json(save_path,values_dict,iteration):
    if os.path.exists(save_path):
        with open(save_path,"r") as f:
            result_json = json.load(f)
        
        
        for index in range(len(result_json["result"])):
            if result_json["result"][index]["iteration"] == iteration:
                result_json["result"][index]["save_values"].append(values_dict)
                optim_values = []
                for each_json in result_json["result"][index]["save_values"]:
                    optim_values.append(each_json["fvalues"])
                mean = np.mean(np.array(optim_values))
                std = np.std(np.array(optim_values))
                result_json["result"][index]["mean"] = mean
                result_json["result"][index]["std"] = std
                with open(save_path,"w") as f2:
                    json.dump(result_json,f2,indent=4)
                return

    else:
        result_json ={
                        "result":[]
                    } 
        
    mean = values_dict["fvalues"]
    std = 0
    each_json = {
                    "iteration":iteration,
                    "save_values":[values_dict],
                    "mean":mean,
                    "std":std
                }

    result_json["result"].append(each_json)
    with open(save_path,"w") as f2:
        json.dump(result_json,f2,indent=4)

def count_files(init_dir,pattern):
    list_dir = os.listdir(init_dir)
    count = 0
    for file_name in list_dir:
        res = re.fullmatch(pattern,file_name)
        if res:
            count +=1
    return count

def find_files(init_dir,pattern):
    list_dir = os.listdir(init_dir)
    output_files = []
    for file_name in list_dir:
        res = re.fullmatch(pattern,file_name)
        if res:
            output_files.append(file_name)
    return output_files
    
if __name__ == "__main__":
    change_json_in_dir("./results")