import paramiko
import os
from paramiko.ssh_exception import BadHostKeyException,BadAuthenticationType,AuthenticationException,SSHException
import stat
from getpass import getpass
from environments import KEYPATH

HOSTNAME = "157.82.22.26"
USERNAME = "u00786"
print("enter your ssh password")
PASSWORD = getpass('your password: ')
SLASH = os.path.join("a","b")[1:-1]

def ssh_connect():
    client = paramiko.client.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy())
    try:
        client.connect(hostname = HOSTNAME,username = USERNAME,password = PASSWORD,key_filename = KEYPATH,timeout = 10)
    except BadHostKeyException:
        print("Bad Host key")
    except BadAuthenticationType:
        print("Bad Authentication")
    except SSHException:
        print("SSH exception")
    except AuthenticationException:
        print("Authentication Exception")
    return client

def check_extension(file_name,extension):
    return file_name.lower().endswith(extension.lower())

def sftp_download_dir(remote_file,local_file,extension = ""):
    with ssh_connect() as client:
        with client.open_sftp() as sftp:
            file_infos = sftp.listdir_attr(remote_file)
            file_list = []
            dir_list = []
            for file_info in file_infos:
                file_name = file_info.filename
                if stat.S_ISDIR(file_info.st_mode):
                    dir_list.append(file_name)
                else:
                    modify_dt = file_info.st_mtime
                    if check_extension(file_name,extension):
                        file_list.append(("",file_name,modify_dt))
            
            while len(dir_list) > 0:
                dir_name = dir_list.pop()
                file_infos = sftp.listdir_attr(remote_file + "/" + dir_name)
                for file_info in file_infos:
                    file_name = file_info.filename
                    if stat.S_ISDIR(file_info.st_mode):
                        dir_list.append(dir_name + "/" + file_name)
                    else:
                        modify_dt = file_info.st_mtime
                        if check_extension(file_name,extension):
                            file_list.append(( dir_name,file_name,modify_dt))
            
            for dir_name,file_name,modify_dt in file_list:
                local_dir_name = dir_name.replace("/",SLASH).replace(":",";")
                os.makedirs(os.path.join(local_file,local_dir_name),exist_ok=True)
                log_path = os.path.join(local_file,local_dir_name,file_name+".txt")
                if os.path.exists(log_path):
                    with open(log_path,"r") as f:
                        last_modify_dt = int(f.read())
                    
                    if modify_dt > last_modify_dt:
                        print(file_name)
                        sftp.get(remote_file + "/" + dir_name + "/" + file_name,os.path.join(local_file,local_dir_name,file_name)) 
                        with open(log_path,"w") as f:
                            f.write(str(modify_dt))
                else:
                    print(os.path.join(local_file,local_dir_name,file_name))
                    sftp.get(remote_file + "/" + dir_name + "/" + file_name,os.path.join(local_file,local_dir_name,file_name)) 
                    with open(log_path,"w") as f:
                        f.write(str(modify_dt))

def sftp_download_file(remote_file,local_file):
    with ssh_connect() as client:
        with client.open_sftp() as sftp:
            sftp.get(remote_file,local_file)
            sftp.close()

if __name__ == "__main__":
    REMOTERESULTPATH = "./Research/optimization/results"
    sftp_download_dir(REMOTERESULTPATH,local_file="./results",extension=".json")