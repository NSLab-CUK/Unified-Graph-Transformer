import pandas as pd
import os
import time


def run_training(db_name):
    cmd = f"python exp.py --dataset {db_name} --epochs 1"

    os.system(cmd)

def main():
    #testing 10%
    #db_list= ["brazil", "europe", "usa",'cornell', 'texas', 'wisconsin', "chameleon", "squirrel", "film","bgp",'cora', 'citeseer', 'pubmed']
    db_list= ['pubmed']
    for i in range(len(db_list)):
        db_name = db_list[i]
        run_training(db_name)
        time.sleep(5)
if __name__ == '__main__':
    main()