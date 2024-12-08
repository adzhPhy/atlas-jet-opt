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

def getEtaHistogram(filename, data):
    plt.figure()
    numHist, bin_edges, _ = plt.hist(ak.flatten(data)["eta"], weights=ak.flatten(data)["weight"],alpha=0.5, bins=100, label="Matched Truth Jet eta", color="blue", histtype="step")
    denomHist, bin_edges, _ = plt.hist(ak.flatten(data)["eta"], alpha=0.5, bins=100, label="Truth Jet eta", color="red", histtype="step")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Truth and Matched Truth Jets over eta")
    plt.legend()
    plt.savefig(f'{filename}_truthVsMatchETA.png')
    
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