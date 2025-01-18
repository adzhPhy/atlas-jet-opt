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

def fillHist(file, histogram):
    for batch in uproot.iterate(f"{file}:JetConstituentTree", ['jet_pt'], step_size=100000):
        eventData = batch["jet_pt"]/1000
        filteredData = eventData[ak.any(eventData!=0, axis=1)].tolist()
        histogram.fill([max(event) for event in filteredData])


def getHist(file_list, histogram):
    for file in file_list:
        fillHist(file, histogram)
        print(f"{file} ran")

# Set up the plot with two subplots (one for stacked histograms and one for the ratio)
fig = plt.figure(figsize=(10, 6))

# Define a gridspec with 1 row and 2 columns
gs = GridSpec(2, 1) 
# Create the subplots
ax1 = fig.add_subplot(gs[0])  # Larger subplot
ax2 = fig.add_subplot(gs[1])  # Smaller subplot

s = time.time()
getHist(ufoFiles, h_ufo)
# getHist(pfoFiles, h_pfo)
getHist(pfo2Files, h_pfo_jz2)
e = time.time()
print(f"{e-s:.2f} seconds for datasets")
# Plot stacked histograms
h_ufo.plot1d(ax=ax1, label="UFO+CSSK", color="blue", stack=True, alpha=0.6, linestyle="--")
h_pfo_jz2.plot1d(ax=ax1, label="PFlow_JZ2", color="red", stack=True, alpha=0.6)
ax1.set_xlabel('Value [GeV]')
ax1.set_ylabel('Number of Events')
ax1.set_yscale("log")
ax1.set_title('Distribution of Leading Reconstructed Jets (pT)')
ax1.legend()

# Calculate the ratio of the histograms (avoid division by zero)
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = h_ufo.values() / h_pfo_jz2.values()

# # Plot the ratio
ax2.plot(ratio, label='Ratio (UFO+CSSK / PFlow_JZ2)', color='purple')
ax2.set_xlabel('Value [GeV]')
ax2.set_ylabel('Ratio')
ax2.set_yscale("log")
ax2.axhline(1, color='gray', linestyle='--', linewidth=1)  # Add horizontal line at ratio=1
ax2.legend()

# Show the plots
plt.tight_layout()
plt.savefig("ufocssk_pflowjz2_leadingJetPtDist.png")
plt.close()