
import numpy as np
import os

import utils.utills as imp 
import utils.nodeedgesdata as xtra
import utils.preprocessdata as prep

NUM_JOBS = 1
JOB_ARRAY_NUMBER = 0 


df = prep.Dataset()

df = df.iloc[::-1]
splits = np.array_split(df, NUM_JOBS)


def preprocess(row):
    """
    df = Dataset()
    row = df.iloc[180189] 
    row = df.iloc[177860]  
    preprocess(row)
    """
    savedir_before = imp.get_dir(imp.processed_dir() / row["dataset"] / "before")
    savedir_after = imp.get_dir(imp.processed_dir() / row["dataset"] / "after")

    fpath1 = savedir_before / f"{row['id']}.java"
    with open(fpath1, "w") as f:
        f.write(row["before"])
    fpath2 = savedir_after / f"{row['id']}.java"
    if len(row["diff"]) > 0:
        with open(fpath2, "w") as f:
            f.write(row["after"])
 
    if not os.path.exists(f"{fpath1}.edges.json"):
        xtra.full_run_joern(fpath1, verbose=3)

    if not os.path.exists(f"{fpath2}.edges.json") and len(row["diff"]) > 0:
        xtra.full_run_joern(fpath2, verbose=3)
    
        

if __name__ == "__main__":
    imp.dfmp(splits[JOB_ARRAY_NUMBER], preprocess, ordr=False, workers=8)
