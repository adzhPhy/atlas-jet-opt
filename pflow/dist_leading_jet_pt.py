import uproot 
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import time
import hist

files = [f"user.tapark.41935133._00000{x}.tree.root" for x in range(1,10)] + [f"user.tapark.41935133._00001{x}.tree.root" for x in range(0,10)] + [f"user.tapark.41935133._00002{x}.tree.root" for x in range(0,10)] + [f"user.tapark.41935133._00003{x}.tree.root" for x in range(0,10)] + [f"user.tapark.41935133._00004{x}.tree.root" for x in range(0,5)] 

plt.figure()
plt.xlabel("Value [GeV]")
plt.ylabel("Number of Events")
plt.yscale("log")
plt.title("Distribution of Leading Reconstructed Jets (pT)")

h1 = hist.Hist(hist.axis.Regular(100, 0, 450))

def getDist(file, histogram):
    for batch in uproot.iterate(f"{file}:JetConstituentTree", ['jet_pt'], step_size=100000):
        eventData = batch["jet_pt"]/1000
        filteredData = eventData[ak.any(eventData!=0, axis=1)].tolist()
        histogram.fill([max(event) for event in filteredData])

s1 = time.time()
for file in files:
    s = time.time()
    getDist(file, h1)
    e = time.time()
    print(f"{e-s:.2f} seconds for {file}")

h1.plot1d()
# plt.hist(h1.values(), bins=100, color='blue', histtype="step")
plt.savefig('PFlow_leadingJetDistPT.png')
e1 = time.time()
print(f"Data Set PFlow done in {e1-s1:.2f} seconds")