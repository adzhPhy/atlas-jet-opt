import uproot 
import awkward as ak
import numpy as np
import time
import numba as nb
from matching import graphs

files = ["user.tapark.41935082._000001.tree.root"] 
columns = ['truthJet_pt', 'truthJet_eta', 'truthJet_phi', 'truthJet_m', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_m']

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

def readfiles(filename):
    print("Reading data...")
    read_s = time.time()
    file = uproot.open(filename)
    tree = file["JetConstituentTree"]
    truth_jets_data = tree.arrays(['truthJet_pt','truthJet_eta', 'truthJet_phi'], library="ak")
    reco_jets_data = tree.arrays(['jet_eta', 'jet_phi','jet_pt'], library="ak")
    res = []
        
    print("---- Running matching... ----")
    s = time.time()
    zipd_t = ak.zip({"phi": truth_jets_data["truthJet_phi"], "eta": truth_jets_data["truthJet_eta"]})
    zipd_t = ak.values_astype(zipd_t, "float32")
        
    zipd_r = ak.zip({"phi": reco_jets_data["jet_phi"], "eta": reco_jets_data["jet_eta"]})
    zipd_r = ak.values_astype(zipd_r, "float32")
    for x,y in zip(zipd_t, zipd_r):
        res.append(calcDeltaR(ak.to_numpy(x),ak.to_numpy(y)))
    
    e = time.time()
    print(f"{e-s:.2f} seconds for matching")
    data_pt_eta_weights = ak.zip({"pt":truth_jets_data["truthJet_pt"]/1000, "eta": truth_jets_data["truthJet_eta"], "weight": res})
    return data_pt_eta_weights

for f in files:
    x = readfiles(f)
    # by convention set as first part of string the name of the MC algorithm
    graphs.getPtHistogram(f"ufocssk_{f}",x)
    graphs.getEtaHistogram(f"ufocssk_{f}",x)