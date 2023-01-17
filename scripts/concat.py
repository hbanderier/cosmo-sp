#!/usr/bin/env python3

import os
import pickle as pkl
import time as timer
from nco import Nco
import numpy as np
import click
import logging

# Piz Daint paths for NCO and CDO (cdo is not useful, could be removed)
os.environ["CDO"] = "/project/pr133/hbanderi/miniconda3/envs/env/bin/cdo"
os.environ["NCOpath"] = "/apps/daint/UES/jenkins/7.0.UP03/21.09/daint-gpu/software/NCO/5.0.4-CrayGNU-21.09/bin"
os.environ["PATH"] += ":/apps/daint/UES/jenkins/7.0.UP03/21.09/daint-gpu/software/NCO/5.0.4-CrayGNU-21.09/bin"
PATHBASE = "/scratch/snx3000/hbanderi/data"

@click.command()
@click.argument("varname")
@click.option("--ana", type=click.Choice(["main", "sensi"], case_sensitive=True), help="Which of the two analyses to perform")
def main(varname, ana):
    logging.basicConfig(filename=f'/users/hbanderi/cosmo-sp/scripts/logs/concat_{ana}/{varname}.log', encoding='utf-8', level=logging.DEBUG)
    nco = Nco()
    with open(f"{PATHBASE}/big{ana}/metadata.pickle", "rb") as handle:
        metadata = pkl.load(handle)
    stuff = metadata["variablemap"][varname]
    files_to_load = metadata["files_to_load"]
    comps = metadata["comps"]
    n_mem = metadata["n_mem"]
    anaprefix = "big" if n_mem==100 else ""
    for filename in files_to_load:
        date = filename.lstrip("lffdm").rstrip(".nc")
        ofile = f"{PATHBASE}/{anaprefix}{ana}/{varname}{date}.nc"
        if not os.path.isfile(ofile): # if a crash happened, don't work on the files already done. Runs of ncecat do not produce a file named "ofile" until they are totally done
            t = timer.time()
            ifiles = [f"{PATHBASE}/{comp}/{str(i).zfill(4)}/{stuff[0]}/lffdm{date}.nc" for i in range(n_mem) for comp in comps[ana]] # all ensembles and all members go to the first dim
            options = ["--hpss", "-u member", f"-v {stuff[1]}"] # Only do one variable
            if stuff[0][-4:] == "plev":
                options.append(f"-d pressure,{stuff[2]},{stuff[2]}") # If variable is stored on multiple plev, select relevant plev. Index was defined in the metadata 
            elif stuff[0][-4:] == "zlev":
                options.append(f"-d altitude,{stuff[2]},{stuff[2]}") # Same for zlev
            elif varname[2:4] == "so":
                 options.append(f"-d soil1,{stuff[2]},{stuff[2]}") # And same for soil level
            nco.ncecat(options=options, input=ifiles, output=ofile) # Do the do
            logging.debug(f"Concatenating {len(ifiles)} files (for {varname}, {date}) took {timer.time() - t}s") # Log
            
        
if __name__ == "__main__":
    main()
