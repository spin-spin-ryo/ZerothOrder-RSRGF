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
    time_values = []
    fvalues_std = []
    labeledflag = False
    labeled = {}
    start = 0
    end = -1
    mode = "best"
    xscale = ""
    yscale = ""
    full_line = 100
    for target_path in target_pathes:
        labeled[target_path] = target_path
    
        
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
        if k == "mode":
            mode = v
        if k == "label":
            labeledflag = v
            for target_path in target_pathes:
                if v:
                    solver_name = target_path.split(SLASH)[-2]
                    solver_param_dir = modify_dir_name(target_path.split(SLASH)[-1])
                    temp_solver_params = solver_param_dir.split("_")
                    solver_params = {}
                    for params in temp_solver_params:
                        param,value = params.split(";")
                        solver_params[param] = value
                    try:
                        use_params ={"reduced dim":r"$d={}$".format(solver_params["reduced dim"])}
                    except:
                        use_params = {}
                    param_str = ""
                    if len(use_params) != 0:
                        param_str = " (" + use_params["reduced dim"] + ")"
                    labeled[target_path] = solver_name + param_str
                
    

    for target_path in target_pathes:
        if mode == "best":
            fvalues_files = find_files(target_path,r"fvalues.*\.pth")
            best_file_name = fvalues_files[0]
            min_value = torch.min(torch.load(os.path.join(target_path,best_file_name)))
            for f in fvalues_files[1:]:
                __temp__tensor = torch.load(os.path.join(target_path,f))
                temp_min_value = torch.min(__temp__tensor)
                if temp_min_value == 0:
                    temp_min_value = torch.min(__temp__tensor[__temp__tensor>0])
                if temp_min_value < min_value:
                    min_value = temp_min_value
                    best_file_name = f
            fvalues.append(torch.load(os.path.join(target_path,best_file_name)))
            time_file_name = get_count_from_filename(best_file_name)
            time_values.append(torch.load(os.path.join(target_path,time_file_name)))
        elif mode == "mean":
            fvalues_files = find_files(target_path,r"fvalues.*\.pth")
            if end == -1:
                max_length = 0
                count = 0
                sum_value = None
                
                for fvalue_file in fvalues_files:
                    temp_fvalue = torch.load(os.path.join(target_path,fvalue_file))
                    max_length = max(temp_fvalue.shape[0],max_length)
                
                for fvalue_file in fvalues_files:
                    temp_fvalue = torch.load(os.path.join(target_path,fvalue_file))
                    if temp_fvalue.shape[0] == max_length:
                        count += 1
                        if sum_value is None:
                            sum_value = temp_fvalue
                        else:
                            sum_value += temp_fvalue
                fvalues.append(sum_value/count)
                print(count)
            else:
                count = 0
                sum_value = None
                for fvalue_file in fvalues_files:
                    temp_fvalue = torch.load(os.path.join(target_path,fvalue_file))
                    if temp_fvalue[start:end].shape[0] == end - start:
                        count += 1
                        if sum_value is None:
                            sum_value = temp_fvalue
                        else:
                            sum_value += temp_fvalue
                fvalues.append(sum_value/count)
        
        elif mode == "mean std":
            fvalues_files = find_files(target_path,r"fvalues.*\.pth")
            if end == -1:
                max_length = 0
                sum_values = None
                
                for fvalue_file in fvalues_files:
                    temp_fvalue = torch.load(os.path.join(target_path,fvalue_file))
                    max_length = max(temp_fvalue.shape[0],max_length)
                
                for fvalue_file in fvalues_files:
                    temp_fvalue = torch.load(os.path.join(target_path,fvalue_file))
                    if temp_fvalue.shape[0] == max_length:
                        if sum_values is None:
                            sum_values = temp_fvalue.unsqueeze(0)
                        else:
                            sum_values = torch.cat((sum_values,temp_fvalue.unsqueeze(0)),dim = 0) 
                        
                fvalue_mean = torch.mean(sum_values,dim = 0)
                fvalue_std = torch.std(sum_values,dim = 0)
                fvalues.append(fvalue_mean)
                fvalues_std.append(fvalue_std)
            else:
                sum_values = None
                for fvalue_file in fvalues_files:
                    temp_fvalue = torch.load(os.path.join(target_path,fvalue_file))
                    if temp_fvalue[start:end].shape[0] == end - start:
                        if sum_values is None:
                            sum_values = temp_fvalue.unsqueeze(0)
                        else:
                            sum_values = torch.cat((sum_values,temp_fvalue.unsqueeze(0)),dim = 0) 
                        
                fvalue_mean = torch.mean(sum_values,dim = 0)
                fvalue_std = torch.std(sum_values,dim = 0)
                fvalues.append(fvalue_mean)
                fvalues_std.append(fvalue_std)
    if "time" in xscale:
        for index,(p,v,t) in enumerate(zip(target_pathes,fvalues,time_values)):
            print(p)
            # start = 0　想定
            if end != -1:
                index = t < end
            else:
                index = torch.ones(len(t)).to(torch.bool)
            if "proposed" in p:
                plt.plot(t[index][::full_line],v[index][::full_line],label = labeled[p])
            else:
                plt.plot(t[index][::full_line],v[index][::full_line],label = labeled[p],linestyle = "dotted")
        plt.xlabel("Time[s]")
    
    else:
        for index,(p,v) in enumerate(zip(target_pathes,fvalues)):
            print(p)
            if "proposed" in p:
                plt.plot(np.arange(len(v))[start:end][::full_line],v[start:end][::full_line],label = labeled[p])
            else:
                plt.plot(np.arange(len(v))[start:end][::full_line],v[start:end][::full_line],label = labeled[p],linestyle = "dotted")
            if mode == "mean std":
                plt.fill_between(np.arange(len(v))[start:end][::full_line],v[start:end][::full_line] + fvalues_std[index][start:end][::full_line],v[start:end][::full_line] - fvalues_std[index][start:end][::full_line],alpha = 0.15)
        plt.xlabel("Iterations")
    
    
    if not labeledflag:
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=1,borderaxespad=0)
    else:
        plt.legend()
    plt.ylabel(r"f(x)")
    if "log" in xscale:
        plt.xscale("log")
    if yscale == "log":
        plt.yscale("log")
    plt.show()
    
def add_name_all_dirs(add_char,init_dir,check = True):
    list_dirs = os.listdir(init_dir)
    for dir_name in list_dirs:
        if add_char not in dir_name:
            print(dir_name)
            print(dir_name+add_char)
            if not check:
                os.rename(init_dir+"/"+dir_name,init_dir+"/"+dir_name+add_char)

def get_count_from_filename(file_name):
    # 仕様が悪い 入力はfvalues{}.pth
    count = file_name.replace("fvalues","")
    return "time_values"+count