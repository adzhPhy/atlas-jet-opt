import uproot 
import awkward as ak
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import hist

filesUFO = [f"user.tapark.41935082._00000{x}.tree.root" for x in range(1,10)] + [f"user.tapark.41935082._00001{x}.tree.root" for x in range(0,10)] + [f"user.tapark.41935082._00002{x}.tree.root" for x in range(0,8)] + ["user.tapark.41935082._000029.tree.root", "user.tapark.41935082._000030.tree.root"] 
filesPFO = [f"user.tapark.41935133._00000{x}.tree.root" for x in range(1,10)] + [f"user.tapark.41935133._00001{x}.tree.root" for x in range(0,10)] + [f"user.tapark.41935133._00002{x}.tree.root" for x in range(0,10)] + [f"user.tapark.41935133._00003{x}.tree.root" for x in range(0,10)] + [f"user.tapark.41935133._00004{x}.tree.root" for x in range(0,5)] 
filesPFO2 = [f"user.tapark.42500064._00000{x}.tree.root" for x in range(1,10)] + [f"user.tapark.42500064._00001{x}.tree.root" for x in range(0,2)]

ufoFiles = ["./ufocssk/"+x for x in filesUFO]
pfoFiles = ["./pflow/"+x for x in filesPFO]
pfo2Files = ["./PFlow_JZ2/"+x for x in filesPFO2]

h_ufo = hist.Hist(hist.axis.Regular(100, 0, 450, label="Value [GeV]"))
h_pfo = hist.Hist(hist.axis.Regular(100, 0, 450, label="Value [GeV]"))
h_pfo_jz2 = hist.Hist(hist.axis.Regular(100, 0, 450, label="Value [GeV]"))

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

@nb.njit
def matchingAlgorithm(dataArray1,dataArray2):
    res = [ calcDeltaR(x,y) for x,y in zip(dataArray1,dataArray2) ]
    return res


def fillHist(file, histogram):
    for batch in uproot.iterate(f"{file}:JetConstituentTree", ['truthJet_pt','truthJet_eta', 'truthJet_phi','jet_eta', 'jet_phi','jet_pt'], step_size=100000):
        cut = ak.any(batch["truthJet_pt"]!=0, axis=1)
        eventData = batch[cut]
        zipd_t = ak.zip({"phi": eventData["truthJet_phi"], "eta": eventData["truthJet_eta"], "pt": eventData["truthJet_pt"]/1000})
        zipd_t = ak.values_astype(zipd_t, "float32")        
        zipd_r = ak.zip({"phi": eventData["jet_phi"], "eta": eventData["jet_eta"], "pt": eventData["jet_pt"]/1000})
        zipd_r = ak.values_astype(zipd_r, "float32")
        
        s = time.time()
        weights = matchingAlgorithm(zipd_t, zipd_r)
        e = time.time()
        print(f"--- {e-s:.2f} for matching {file} ---")
        histogram.fill(eventData["jet_pt"], weight=weights)

def getHist(file_list, histogram):
    for file in file_list:
        fillHist(file, histogram)
        print(f"{file} ran")
            
fig = plt.figure(figsize=(10, 6))

# Define a gridspec with 1 row and 2 columns
gs = GridSpec(2, 1) 
# Create the subplots
ax1 = fig.add_subplot(gs[0])  # Larger subplot
ax2 = fig.add_subplot(gs[1])  # Smaller subplot

s = time.time()
getHist(ufoFiles, h_ufo)
getHist(pfoFiles, h_pfo)
# getHist(pfo2Files, h_pfo_jz2)
e = time.time()
print(f"{e-s:.2f} seconds for datasets")

# Plot stacked histograms
h_ufo.plot1d(ax=ax1, label="UFO+CSSK", color="blue", stack=True, alpha=0.6, linestyle="--")
h_pfo.plot1d(ax=ax1, label="PFlow", color="red", stack=True, alpha=0.6)
ax1.set_xlabel('Value [GeV]')
ax1.set_ylabel('Number of Jets')
ax1.set_yscale("log")
ax1.set_title('Distribution of Matched Reconstructed Jets (pT)')
ax1.legend()

# Calculate the ratio of the histograms (avoid division by zero)
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = h_ufo.values() / h_pfo_jz2.values()

# # Plot the ratio
ax2.plot(ratio, label='Ratio (UFO+CSSK / PFlow)', color='purple')
ax2.set_xlabel('Value [GeV]')
ax2.set_ylabel('Ratio')
ax2.axhline(1, color='gray', linestyle='--', linewidth=1)  # Add horizontal line at ratio=1
ax2.legend()

# Show the plots
plt.tight_layout()
plt.savefig("ufocssk_pflow_truthJetIsReco.png")
plt.close()