import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

def getPtHistogram(filename, data):
    plt.figure()
    numHist, bin_edges, _ = plt.hist(ak.flatten(data)["pt"], weights=ak.flatten(data)["weight"],alpha=0.5, bins=100, label="Matched Truth Jet pT", color="blue", histtype="step")
    denomHist, bin_edges, _ = plt.hist(ak.flatten(data)["pt"], alpha=0.5, bins=100, label="Truth Jet pT", color="red", histtype="step")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.title("Distribution of Truth and Matched Truth Jets over pT")
    plt.legend()
    plt.savefig(f'{filename}_truthVsMatchPT.png')
    plt.close()
    
    plt.figure()
    efficiency = np.divide(numHist, denomHist, out=np.zeros_like(numHist), where=denomHist != 0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.step(bin_centers ,efficiency, where="mid", label="Efficiency", color="green")
    plt.xlabel("Value")
    plt.ylabel("Efficiency")
    plt.title("Efficiency of Matching Truth Jets over pT")
    plt.ylim(0, 1.2)
    plt.legend()
    plt.savefig(f'{filename}_efficiencyPT.png')
    plt.close()

def getEtaHistogram(filename, data):
    plt.figure()
    numHist, bin_edges, _ = plt.hist(ak.flatten(data)["eta"], weights=ak.flatten(data)["weight"],alpha=0.5, bins=100, label="Matched Truth Jet eta", color="blue", histtype="step")
    denomHist, bin_edges, _ = plt.hist(ak.flatten(data)["eta"], alpha=0.5, bins=100, label="Truth Jet eta", color="red", histtype="step")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Truth and Matched Truth Jets over eta")
    plt.legend()
    plt.savefig(f'{filename}_truthVsMatchETA.png')
    plt.close()
    
    plt.figure()
    efficiency = np.divide(numHist, denomHist, out=np.zeros_like(numHist), where=denomHist != 0)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.step(bin_centers ,efficiency, where="mid", label="Efficiency", color="green")
    plt.xlabel("Value")
    plt.ylabel("Efficiency")
    plt.title("Efficiency of Matching Truth Jets over eta")
    plt.ylim(0, 1.2)
    plt.legend()
    plt.savefig(f'{filename}_efficiencyETA.png')
    plt.close()

def getEnergyResponseHistogram(filename, dataUFO, dataPFO):
    tempbins = [0.2, 0.7, 1.0, 1.3, 1.8, 2.5, 2.8, 3.2, 3.5, 4.5]

    bin_edges = tempbins
    total_range = bin_edges[-1] - bin_edges[0]
    bin_edges = np.linspace(bin_edges[0], bin_edges[-1], 101)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure()
    countsUFO, _, _ = plt.hist(ak.flatten(dataUFO), bins=bin_edges,alpha=0.6, color="green", histtype="step")
    countsPFO, _, _ = plt.hist(ak.flatten(dataPFO), bins=bin_edges, alpha=0.6, color="purple", histtype="step")
    
    mean_ufo = np.average(bin_centers, weights=countsUFO)
    variance_ufo = np.average((bin_centers - mean_ufo) ** 2, weights=countsUFO)
    std_dev_ufo = np.sqrt(variance_ufo)
    
    mean_pfo = np.average(bin_centers, weights=countsPFO)
    variance_pfo = np.average((bin_centers - mean_pfo) ** 2, weights=countsPFO)
    std_dev_pfo = np.sqrt(variance_pfo)
    
    plt.title("Energy Response Distribution UFO+CSSK vs EMPflow")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xticks(tempbins)
    plt.legend([f"UFO+CSSK Mean: {mean_ufo:.2f}, Std Dev: {std_dev_ufo:.2f}", f"EMPFlow Mean: {mean_pfo:.2f}, Std Dev: {std_dev_pfo:.2f}"], loc='upper right')
    
    plt.savefig(f"{filename}_energyResp")
    plt.close()