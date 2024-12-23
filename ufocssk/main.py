import uproot 
import awkward as ak
import numpy as np
import numba as nb
from matching import graphs
import time
import os

files = [f"user.tapark.41935082._00001{x}.tree.root" for x in range(0,10)] + [f"user.tapark.41935082._00002{x}.tree.root" for x in range(0,8)] + ["user.tapark.41935082._000029.tree.root", "user.tapark.41935082._000030.tree.root"] 

@nb.njit
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

def getData(filename):
    print(f"--- Reading {filename} ... ---")
    file = uproot.open(filename)
    tree = file["JetConstituentTree"]
    data = tree.arrays(['truthJet_pt','truthJet_eta', 'truthJet_phi','jet_eta', 'jet_phi','jet_pt'], library="ak")
    # remove events with any empty arrays
    pts = data["truthJet_pt"]
    cut = ak.any(pts!=0, axis=1)
    eventData = data[cut]
    zipd_t = ak.zip({"phi": eventData["truthJet_phi"], "eta": eventData["truthJet_eta"], "pt": eventData["truthJet_pt"]/1000 })
    zipd_t = ak.values_astype(zipd_t, "float32")        
    zipd_r = ak.zip({"phi": eventData["jet_phi"], "eta": eventData["jet_eta"], "pt": eventData["jet_pt"]/1000 })
    zipd_r = ak.values_astype(zipd_r, "float32")
    return zipd_t, zipd_r

@nb.njit
def matchingAlgorithm(dataArray1,dataArray2):
    res = [ calcDeltaR(x,y) for x,y in zip(dataArray1,dataArray2) ]
    return res

def calcMatching(filename):
    data_t, data_r = getData(filename)
    s = time.time()
    weights = matchingAlgorithm(data_t, data_r)
    e = time.time()
    print(f"--- {e-s:.2f} for matching {filename} ---")
    data = ak.zip({"pt": data_t["pt"], "eta": data_t["eta"], "weight": weights})
    return data

data = calcMatching(filenameUFO)
graphs.getPtHistogram(filenameUFO, data)

process_start = time.time()
for file in files:
    data = calcMatching(file)
    # extFile = uproot.update(file)
    # extFile.mktree("JetXIsReco", {"truthJetXIsReco": "var*int64"})
    # extFile["JetXIsReco"].extend({"truthJetXIsReco": data})
    # by convention set as first part of string the name of the Reconstruction algorithm
    # graphs.getPtHistogram(f"ufocssk_{f}",data)
    # graphs.getEtaHistogram(f"ufocssk_{f}",data)
process_end = time.time()
print("----------------------------------")
print(f"CSSK dataset took {process_end-process_start:.2f} seconds")