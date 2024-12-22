import uproot 
import awkward as ak
import numpy as np
import time
import numba as nb
from matching import graphs

ufoFileNames  =  [f"user.tapark.41935082._00001{x}.tree.root" for x in range(2,5)]
pfoFileNames = [f"user.tapark.41935133._00001{x}.tree.root" for x in range(2,5)] 

ufoFiles = ["./ufocssk/"+x for x in ufoFileNames]
pfoFiles = ["./pflow/"+x for x in pfoFileNames]

def getData(filename):
    print(f"Filtering {filename} ...")
    s = time.time()
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
    e = time.time()
    print(f"{e-s:.2f} seconds for data set {filename}")
    return zipd_t, zipd_r

@nb.njit
def truthJetMatching(jetTArray, jetR):
    matched_t = []
    for j in range(len(jetTArray)):
        dphi = abs(jetR["phi"] - jetTArray["phi"][j])
        dphi = dphi if dphi <= np.pi else 2*np.pi - dphi # take into account periodicity
        deta = jetR["eta"] - jetTArray["eta"][j]
        delta_r = np.sqrt(dphi**2+deta**2)
        if delta_r < 0.4:
            matched_t.append(jetTArray[j])
        else:
            continue   
    return matched_t

@nb.njit
def recoJetNearby(ind, jetRArray):
    # ind is the index of the initial jet
    isNotIsolated = []
    for i in range(len(jetRArray)):
        if i==ind:
            continue
        else:
            dphi = abs(jetRArray["phi"][ind] - jetRArray["phi"][i])
            dphi = dphi if dphi <= np.pi else 2*np.pi - dphi # take into account periodicity
            deta = jetRArray["eta"][ind] - jetRArray["eta"][i]
            delta_r = np.sqrt(dphi**2+deta**2)
            if delta_r < 0.6:
                isNotIsolated.append(np.True_)
            else:
                continue
    if np.True_ in isNotIsolated:
        return np.True_
    else:
        return np.False_

@nb.njit
def calcEnergyResp(trueJetArray, recoJetArray):    
    data = []
    for i in range(len(recoJetArray)):
        matched_truth_jets = truthJetMatching(trueJetArray, recoJetArray[i])
        if len(matched_truth_jets)==0:
            continue
        else:
            max_pt_truth_jet = np.sort([jet["pt"] for jet in matched_truth_jets])[-1]
            if recoJetNearby(i, recoJetArray):
                continue
            else:
                data.append(recoJetArray["pt"][i]/max_pt_truth_jet)
    return data

@nb.njit
def getEnergyResponse(a,b):
    data = [calcEnergyResp(x,y) for x,y in zip(a,b)]
    return data

for x,y in zip(ufoFiles, pfoFiles):
    dataUFO_t, dataUFO_r = getData(x)
    dataPFO_t, dataPFO_r = getData(y)
    print(f"Calculating Energy Response for {x}...")
    s1 = time.time()
    dataUFO = getEnergyResponse(dataUFO_t,dataUFO_r)
    e1 = time.time()
    print(f"{e1-s1:.2f} seconds for calculation")
    print(f"Calculating Energy Response for {y}...")
    s2 = time.time()
    dataPFO = getEnergyResponse(dataPFO_t,dataPFO_r)
    e2 = time.time()
    print(f"{e2-s2:.2f} seconds for calculation")
    filenameUFO = x.split(".")[-3]
    graphs.getEnergyResponseHistogram(filenameUFO,dataUFO,dataPFO)
    