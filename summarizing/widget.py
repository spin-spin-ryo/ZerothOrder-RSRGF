
import tkinter as tk
import tkinter.ttk as ttk
import os
from summarizing.utils import get_dir_from_dict,get_best_result_path

class prop_cmbbox:
    def __init__(self,parent,values,state,key,index):
        self.cmbbox = ttk.Combobox(parent,values=values,state=state)
        self.index = index
        self.key = key
    
    def func(self,event):
        print(self.cmbbox.get())
    
    def check(self,prop_dict):
        all_ok = True
        for k,v in prop_dict.items():
            if v == "":
                all_ok = False
        return all_ok
    
    def bind(self):
        self.cmbbox.bind("<<ComboboxSelected>>",self.func)
    
    def grid(self,row,column):
        self.cmbbox.grid(row = row,column = column)


class solver_prop_cmbbox(prop_cmbbox):
    def bind(self,prop_dict):
        __func__ = lambda event:self.func(prop_dict)
        self.cmbbox.bind("<<ComboboxSelected>>",__func__)
    
    def func(self,prop_dict):
        prop_dict[self.key] = self.cmbbox.get()


    
class page:
    def __init__(self,notebook,solver_dirs,solver_prop_dict) -> None:
        self.main_frame = tk.Frame(notebook)
        self.solver_cmbbox = ttk.Combobox(self.main_frame,values = solver_dirs,state="readonly")
        self.solver_cmbbox.bind("<<ComboboxSelected>>",self.generate_cmbboxes)
        self.solver_cmbbox.pack()
        self.solver_prop_dict = solver_prop_dict
        self.comboxes = []
        self.prop_dict = {}
        self.path = ""
        self.flag = False
    
    def add(self,notebook,text):
        notebook.add(self.main_frame,text = text)
    
    def generate_cmbboxes(self,event):
        if self.flag:
            return
        self.flag = True
        sub_frame = tk.Frame(self.main_frame)
        solver_name = self.solver_cmbbox.get()
        solver_params = self.solver_prop_dict[solver_name]
        for key in solver_params.keys():
            self.prop_dict[key] = ""
        count = 0
        for k,v in solver_params.items():
            tk.Label(sub_frame,text=k).grid(row = count,column=0)
            cmbbox = solver_prop_cmbbox(sub_frame,values=v,state="readonly",key = k,index = count)
            cmbbox.bind(self.prop_dict)
            cmbbox.grid(row = count,column=1)
            count += 1
        sub_frame.pack()
        return

    def get_path(self,current_path):
        check_flag = True
        for v in self.prop_dict.values():
            if v == "":
                check_flag = False
        if check_flag:
            self.path = os.path.join(self.solver_cmbbox.get(),get_dir_from_dict(self.prop_dict))
        else:
            init_dir = os.path.join(current_path,self.solver_cmbbox.get())
            best_dir,min_val = get_best_result_path(init_dir,self.prop_dict)
            print("最小値:",min_val)
            self.path = os.path.join(self.solver_cmbbox.get(),best_dir)

def generate_entry(parent,text):
    frame = tk.Frame(parent)
    tk.Label(frame,text=text).pack(side = "left")
    entry = tk.Entry(frame)
    entry.pack(side = "left")
    frame.pack()
    return entry
