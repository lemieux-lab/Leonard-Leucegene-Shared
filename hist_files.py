import pandas as pd
import numpy as np 
import os 
import pdb
import matplotlib.pyplot as plt

print("Listing files, one moment ...")
os.system('rm hist_files; ls -l -R RES | grep -e "sauves" > hist_files')
hf = pd.read_csv("hist_files", sep = "\t", skiprows=1).iloc[:,0].str.split("\s*", expand = True)
hf["month"] = hf[5]
hf["day"] = hf[6].astype(int)
hf["time"] = hf[7]
hf["size"] = hf[4].astype(float) / 1e3 # kb to --> megabytes
hf = hf.groupby(["month","day"]).sum().reset_index()
hf["n_month"] = pd.Series([{"Aug": 8, "Sep": 9, "Oct": 10, "Nov":11, "Dec": 12,"Jan": 1}[m] for m in hf["month"] ]).astype(str).str.zfill(2) + "-" + hf["month"]
hf = hf.sort_values(["n_month", "day"])
hf["t"]  = hf["month"] + "-" + hf["day"].astype(str)
fig, ax =plt.subplots(figsize = (13,7))
ax.grid(zorder = 0, linestyle='--',alpha = 0.5)
ax.bar(np.arange(hf.shape[0]), hf["size"], alpha = 0.95, zorder = 3)
for i, z in zip(np.arange(hf.shape[0]), hf["size"]):
    ax.text(i - 0.5, z, "{:0.2f}GB".format(z / 1e3))
ax.set_xticklabels(hf["t"], rotation = 90)
ax.set_title("Size of generated files")
ax.set_xlabel("Month-day")
ax.set_xticks(np.arange(hf.shape[0]))
ax.set_ylabel("Size in mb")
ax.set_yscale("log")
#ax.set_yticklabels(10 ** )
plt.savefig("RES/hist_files.svg")
