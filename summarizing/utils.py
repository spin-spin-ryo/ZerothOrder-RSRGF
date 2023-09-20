import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from summarizing.modify_json import find_files

SLASH = os.path.join("a","b")[1:-1]

def get_dir_from_dict(prop_dict):
    output_path = ""
    for k,v in prop_dict.items():
        output_path += k +";" +v +"_"
    return output_path[:-1]

def modify_dir_name(dir_name,global2local = True):
    change_names = [
        ("reduced_dim","reduced dim"),
        ("sample_size","sample size"),
        ("step_schedule","step schedule")
    ]
    if global2local:
        for a,b in change_names:
            dir_name = dir_name.replace(a,b)
        return dir_name
    else:
        for a,b in change_names:
            dir_name = dir_name.replace(b,a)
        return dir_name


def get_allparams_from_dir(dir_path):
    param_dirs = os.listdir(dir_path)
    default_params = {}
    for dir_name in param_dirs:
        dir_name = modify_dir_name(dir_name)
        if len(default_params) == 0:
            params = dir_name.split("_")
            for param in params:
                print(param)
                key,val = param.split(";")
                default_params[key] = [val]    
        else:
            params = dir_name.split("_")
            for param in params:
                key,val = param.split(";")
                if val not in default_params[key]:
                    default_params[key].append(val)
    return default_params

def get_params_from_dir(dir_name):
    dir_name = modify_dir_name(dir_name)
    default_params = {}
    params = dir_name.split("_")
    for param in params:
        key,val = param.split(";")
        default_params[key] = val
    return default_params    
    

def get_best_result_path(init_dir,prop_dict):
    # init_dir以下でprop_dictで指定されている要素の中から最適解のpathを見つけてくる.
    # init_dirはsolver_nameまでのdirで
    dir_list = os.listdir(init_dir)
    min_val = None
    min_val_dir = None
    for dir_name in dir_list:
        now_prop_dict = get_params_from_dir(dir_name)
        ok_flag = True
        # check
        for k,v in prop_dict.items():
            if v != "" and v != now_prop_dict[k]:
                ok_flag = False
        if ok_flag:
            now_val,_ = get_min_val_from_result(os.path.join(init_dir,dir_name,"result.json"))
            if np.isnan(now_val):
                continue
            if min_val is None:
                min_val = now_val
                min_val_dir = dir_name
            else:
                if now_val < min_val:
                    min_val = now_val
                    min_val_dir = dir_name
    return min_val_dir,min_val


def get_min_val_from_result(file_name):
    with open(file_name) as f:
        result_json= json.load(f)
    
    mean_val = None
    min_val = None
    for each_json in result_json["result"]:
        if mean_val is None:
            mean_val = each_json["mean"]
        else:
            if mean_val > each_json["mean"]:
                mean_val = each_json["mean"]

        for values_dict in each_json["save_values"]:
            if min_val is None:
                min_val = values_dict["fvalues"]
            else:
                if min_val > values_dict["fvalues"]:
                    min_val = values_dict["fvalues"]

    return min_val,mean_val

def modify_local2global(path):
    path = path.replace(";",":")
    path = path.replace(SLASH,"/")
    return path


def plot_result(target_pathes,*args):
    fvalues = []
    for target_path in target_pathes:
        fvalues_files = find_files(target_path,r"fvalues.*\.pth")
        print(fvalues_files)
        best_file_name = fvalues_files[0]
        min_value = torch.min(torch.load(os.path.join(target_path,best_file_name)))
        for f in fvalues_files[1:]:
            temp_min_value = torch.min(torch.load(os.path.join(target_path,f)))
            if temp_min_value < min_value:
                min_value = temp_min_value
                best_file_name = f
        fvalues.append(torch.load(os.path.join(target_path,best_file_name)))
    
    start = 0
    end = -1
    xscale = ""
    yscale = ""
    full_line = 100
    #option関連
    for k,v in args[0].items():
        if k == "start":
            start = v
        if k == "end":
            end = v
        if k == "xscale":
            xscale = v
        if k == "yscale":
            yscale = v
        if k == "full_line":
            full_line = v

    
    for p,v in zip(target_pathes,fvalues):
        print(p)
        if "proposed" in p:
            plt.plot(np.arange(len(v))[start:end][::full_line],v[start:end][::full_line],label = p)
        else:
            plt.plot(np.arange(len(v))[start:end][::full_line],v[start:end][::full_line],label = p,linestyle = "dotted")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=1,borderaxespad=0)
    if xscale != "":
        plt.xscale("log")
    if yscale != "":
        plt.yscale("log")
    plt.show()
    