import paramiko
import os
from paramiko.ssh_exception import BadHostKeyException,BadAuthenticationType,AuthenticationException,SSHException
import stat
from getpass import getpass
from datetime import datetime

HOSTNAME = "157.82.22.26"
USERNAME = "u00786"
print("enter your ssh password")
PASSWORD = getpass('your password: ')
KEYPATH = "C:\\Users\\19991\\.ssh\\ist"

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


def sftp_download_dir(remote_file,local_file,extension = ""):
    with ssh_connect() as client:
        with client.open_sftp() as sftp:
            file_infos = sftp.listdir_attr(remote_file)
            for file_info in file_infos:
                print(file_info.st_mtime)
                print(type(file_info.st_mtime))
                modify_dt = datetime.fromtimestamp(file_info.st_mtime)
                print(modify_dt)
                print(type(modify_dt))
    
    a = int(234567890)

    with open("test.txt","w") as f:
        f.write(str(a))
    
    with open("test.txt","r") as f:
        b = f.read()
        print(int(b))

REMOTERESULTPATH = "./Research/optimization/results"
sftp_download_dir(REMOTERESULTPATH,local_file=".\\results",extension=".json")