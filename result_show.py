import os
import matplotlib.pyplot as plt
import torch
from environments import RESULTPATH

import tkinter as tk
import tkinter.ttk as ttk

from summarizing.widget import prop_cmbbox,solver_prop_cmbbox,page,generate_entry
from summarizing.utils import get_dir_from_dict,get_allparams_from_dir,modify_dir_name,modify_local2global,plot_result
from download_result import sftp_download_dir

REMOTERESULTPATH = "./Research/optimization/results"
    
problem_prop_comboboxes = []
problem_prop_frame =None
problem_prop_dict = {}
CURRENT_PATH = RESULTPATH

solver_prop_comboboxes = []
solver_prop_frame =None
solver_prop_dict = {}

pages = []
buttons = {}
notebook = None
option_entries = {}
option_window = None

def initialize():
    global problem_prop_frame
    global problem_prop_comboboxes
    global problem_prop_dict
    global CURRENT_PATH
    if problem_prop_frame is not None:
        problem_prop_frame.destroy()
    problem_prop_comboboxes = []
    problem_prop_dict = {}
    CURRENT_PATH = RESULTPATH

    global notebook
    if notebook is not None:
        notebook.destroy()
    global pages
    pages = []
    
    global buttons
    for k in buttons.keys():
        buttons[k].destroy()
    buttons = {}

class problem_prop_cmbbox(prop_cmbbox):
    def __init__(self, parent, values, state, key, index):
        super().__init__(parent, values, state, key, index)
    
    def func(self, event):
        global problem_prop_dict
        problem_prop_dict[self.key] = self.cmbbox.get()
        if self.check(problem_prop_dict):
            self.generate_next(problem_prop_dict)
    
    def generate_next(self,prop_dict):
        global CURRENT_PATH
        next_dir = get_dir_from_dict(prop_dict)
        CURRENT_PATH = os.path.join(CURRENT_PATH,next_dir)
        print(CURRENT_PATH)
        solver_dirs = os.listdir(CURRENT_PATH)
        global solver_prop_dict
        for solver in solver_dirs:
            next_path = os.path.join(CURRENT_PATH,solver)
            solver_prop_dict[solver] = get_allparams_from_dir(next_path)
        global root
        global notebook
        notebook = ttk.Notebook(root)
        onepage = page(notebook,solver_dirs,solver_prop_dict)
        onepage.add(notebook,text = "tab1")
        global pages
        global buttons
        pages = [onepage]
        notebook.pack()

        add_button_command = lambda : add_page(notebook,solver_dirs,solver_prop_dict)
        add_button =  tk.Button(root,text = "追加",command=add_button_command)
        add_button.pack()
        buttons["add_button"] = add_button
        remove_button = tk.Button(root,text = "削除", command = remove_button_command)
        remove_button.pack()
        buttons["remove_button"] = remove_button
        option_button = tk.Button(root,text="オプション",command=open_option_window)
        option_button.pack()
        buttons["option_button"] = option_button
        exec_button = tk.Button(root,text = "実行",command=execute_plot)
        exec_button.pack()
        buttons["exec_button"] = exec_button
        return

def open_option_window():
    global root
    global option_window
    if option_window is None:
        option_window = tk.Toplevel(root)
    else:
        option_window.lift()
        return
    global option_entries
    entry_start = generate_entry(option_window,"start")
    entry_end = generate_entry(option_window,"end")
    entry_start.insert(0,"0")
    entry_end.insert(0,"-1")
    entry_xscale = generate_entry(option_window,"xscale")
    entry_yscale = generate_entry(option_window,"yscale")

    option_entries[("start",int)] = entry_start
    option_entries[("end",int)] = entry_end
    option_entries[("xscale",str)] = entry_xscale
    option_entries[("yscale",str)] = entry_yscale

def remove_button_command():
    global pages
    global notebook
    notebook.forget(len(pages)-1)
    pages.pop()


def add_page(notebook,solver_dirs,solver_prop_dict):
    global pages
    page_ =  page(notebook,solver_dirs,solver_prop_dict)
    page_.add(notebook,text = f"tab{len(pages) + 1}")
    pages.append(page_)

def get_options():
    global option_entries
    options = {}
    for k,entry in option_entries.items():
        key_name = k[0]
        key_type = k[1]
        options[key_name] = key_type(entry.get())
    return options

def execute_plot():
    global pages
    
    target_pathes = []
    for each_page in pages:
        each_page.get_path(CURRENT_PATH)
        now_path = each_page.path
        now_path = modify_dir_name(now_path,global2local=False)
        target_path = os.path.join(CURRENT_PATH,now_path)
        problem_dir = CURRENT_PATH[len(RESULTPATH):]
        global_problem_dir = modify_local2global(problem_dir)
        global_path = REMOTERESULTPATH + global_problem_dir +"/"+ modify_local2global(now_path)    
        if now_path != "" and os.path.exists(target_path):
            sftp_download_dir(global_path,target_path,extension=".pth")
            if os.path.exists(os.path.join(target_path,"fvalues.pth")):
                target_pathes.append(target_path)

    option_args = get_options()

    plot_result(target_pathes,option_args)

def get_problem_names():
    prob_lis = os.listdir(RESULTPATH)
    return prob_lis


def generate_problem_properties_box(event):
    global problem_prop_frame
    global problem_prop_comboboxes
    global problem_prop_dict
    global CURRENT_PATH
    initialize()

    problem_prop_frame = tk.Frame(root)
    problem_prop_frame.pack()
    temp_problem_name = problem_name.get()
    prop_dirs = os.listdir(os.path.join(RESULTPATH,temp_problem_name))
    CURRENT_PATH = os.path.join(RESULTPATH,temp_problem_name)
    default_prop_list = {}
    for prop in prop_dirs:
        prop_list = prop.split("_")
        if len(default_prop_list) == 0:
            for each_prop in prop_list:
                key,val = each_prop.split(";")
                default_prop_list[key] = [val]
                problem_prop_dict[key] = ""
        else:
            if len(prop_list) != len(default_prop_list):
                raise ValueError("prop mismatch.")
            for each_prop in prop_list:
                key,val = each_prop.split(";")
                if val not in default_prop_list[key]:
                    default_prop_list[key].append(val)
        
        count = 0
        for k,v in default_prop_list.items():
            tk.Label(problem_prop_frame,text=k).grid(row = count,column=0)
            cmbbox = problem_prop_cmbbox(problem_prop_frame,values = v,state="readonly",key=k,index = count)
            cmbbox.bind()
            cmbbox.grid(count,1)
            problem_prop_comboboxes.append(cmbbox)
            count +=1
            
            

problem_names = get_problem_names()
root = tk.Tk()
root.geometry("300x400")
problem_name = tk.StringVar()
problem_name_box = ttk.Combobox(root,textvariable=problem_name,state = "readonly",values = problem_names)
problem_name_box.bind("<<ComboboxSelected>>",generate_problem_properties_box)
problem_name_box.pack()
root.mainloop()