import uproot 
import awkward as ak
import numpy as np
import numba as nb
from matching import graphs
import time
import os

files = [f"user.tapark.41935082._00001{x}.tree.root" for x in range(3,10)] + [f"user.tapark.41935082._00002{x}.tree.root" for x in range(0,8)] + ["user.tapark.41935082._000029.tree.root", "user.tapark.41935082._000030.tree.root"] 


def get_file_size(filename):
    file_size = os.stat(f'./{filename}').st_size
    return file_size / 1024**2

@nb.njit(cache=True)
def calcDeltaR(a,b):
    res = []
    for i in range(len(a.phi)):
        delta_r_row = []
        for j in range(len(b.phi)):
            dphi = abs(a.phi[i]-b.phi[j])
            dphi = dphi if dphi <= np.pi else 2*np.pi - dphi
            deta = a.eta[i] - b.eta[j]
            delta_r = np.sqrt(dphi**2+deta**2)
            if delta_r < 0.4:
                delta_r_row.append(1)
            else:
                delta_r_row.append(0)
        if 1 in delta_r_row:
            res.append(1)
        else:
            res.append(0)
    return res

def readfiles(filename):
    print(f"Reading {filename}...")
    read_s = time.time()
    print("---- Running matching... ----")
    s = time.time()
    file = uproot.open(filename)
    tree = file["JetConstituentTree"]
    array = tree.arrays(['truthJet_pt','truthJet_eta', 'truthJet_phi', 'jet_eta', 'jet_phi', 'jet_pt'], library="ak")   
    zipd_t = ak.zip({"phi": array["truthJet_phi"], "eta": array["truthJet_eta"]})
    zipd_t = ak.values_astype(zipd_t, "float32")
            
    zipd_r = ak.zip({"phi": array["jet_phi"], "eta": array["jet_eta"]})
    zipd_r = ak.values_astype(zipd_r, "float32")
    res = [calcDeltaR(ak.to_numpy(x),ak.to_numpy(y)) for x,y in zip(zipd_t, zipd_r)]
    e = time.time()
    print(f"{e-s:.2f} seconds for matching of {filename} | {get_file_size(filename):.2f} MB")
    # data_pt_eta_weights = ak.zip({"pt":array["truthJet_pt"]/1000, "eta": array["truthJet_eta"], "weight": res})
    return ak.Array(res)

process_start = time.time()
for file in files:
    x = readfiles(file)
    extFile = uproot.update(file)
    extFile.mktree("JetXIsReco", {"truthJetXIsReco": "var*int64"})
    extFile["JetXIsReco"].extend({"truthJetXIsReco": x})
    # by convention set as first part of string the name of the Reconstruction algorithm
    # graphs.getPtHistogram(f"ufocssk_{f}",x)
    # graphs.getEtaHistogram(f"ufocssk_{f}",x)
process_end = time.time()
print("----------------------------------")
print(f"CSSK dataset took {process_end-process_start:.2f} seconds")