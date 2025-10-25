import sys
import time

sys.path.append("./src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

update_cnt = 10

listMethods = ['iNN', 'NN', 'iODE', 'ODE', 'ENTROPY', 'RANDOM']
resultFolder = './resultsOnLambda_100/'

drawIndex = 0
iNN = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
iNN = iNN.mean(0)
iNNT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
iNNT = iNNT.mean(0)

drawIndex = 1
NN = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
NN = NN.mean(0)
NNT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
NNT = NNT.mean(0)

# drawIndex = 2
# iODE = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
# iODE = iODE.mean(0)
# iODET = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
# iODET = iODET.mean(0)

drawIndex = 3
ODE = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
ODE = ODE.mean(0)
ODET = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
ODET = ODET.mean(0)

drawIndex = 4
ENT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
ENT = ENT.mean(0)
ENTT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
ENTT = ENTT.mean(0)

drawIndex = 5
RND = np.loadtxt(resultFolder + listMethods[drawIndex] + '_MOCU.txt', delimiter = "\t")
RND = RND.mean(0)
RNDT = np.loadtxt(resultFolder + listMethods[drawIndex] + '_timeComplexity.txt', delimiter = "\t")
RNDT = RNDT.mean(0)

x_ax = np.arange(0,update_cnt+1,1)
# plt.plot(x_ax, iNN, 'r*:', x_ax, NN, 'rs--', x_ax, ENT, 'gd:', x_ax, RND, 'b,:')
# plt.legend(['NN-based MOCU (iterative)', 'NN-based MOCU', 'Entropy-based', 'Random'])
plt.plot(x_ax, iNN, 'r*:', x_ax, NN, 'rs--', x_ax, ODE, 'yo--', x_ax, ENT, 'gd:', x_ax, RND, 'b,:')
plt.legend(['Proposed (iterative)', 'Proposed', 'ODE', 'Entropy', 'Random'])
# plt.plot(x_ax, iNN, 'r*:', x_ax, NN, 'rs--', x_ax, iODE, 'yp:', x_ax, ODE, 'yo--', x_ax, ENT, 'gd:', x_ax, RND, 'b,:')
# plt.legend(['NN-based MOCU (iterative)', 'NN-based MOCU', 'ODE-based MOCU (iterative)', 'ODE-based MOCU', 'Entropy-based', 'Random'])
plt.xticks(np.arange(0, update_cnt+1,1)) 
plt.xlabel('Number of updates')
plt.ylabel('MOCU')
# plt.title('Experimental design for N=5 oscillators')
plt.grid(True)
plt.savefig(resultFolder + "MOCU_5.png", dpi=300)

# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: All methods (log scale)
ax1.plot(x_ax, np.insert(np.cumsum(iNNT), 0, 0.0000000001), 'r*:', 
         x_ax, np.insert(np.cumsum(NNT), 0, 0.0000000001), 'rs--', 
         x_ax, np.insert(np.cumsum(ODET), 0, 0.0000000001), 'yo--', 
         x_ax, np.insert(np.cumsum(ENTT), 0, 0.0000000001), 'gd:', 
         x_ax, np.insert(np.cumsum(RNDT), 0, 0.0000000001), 'b,:')
ax1.legend(['Proposed (iterative)', 'Proposed', 'ODE', 'Entropy', 'Random'])
ax1.set_yscale('log')
ax1.set_xlabel('Number of updates')
ax1.set_ylabel('Cumulative time (seconds, log scale)')
ax1.set_xticks(np.arange(0, update_cnt+1, 1))
ax1.set_ylim(1, 10000)
ax1.set_title('All methods comparison')
ax1.grid(True)

# Right plot: Neural network methods only (linear scale)
ax2.plot(x_ax, np.insert(np.cumsum(iNNT), 0, 0.0), 'r*:', 
         x_ax, np.insert(np.cumsum(NNT), 0, 0.0), 'rs--',
         x_ax, np.insert(np.cumsum(ENTT), 0, 0.0), 'gd:', 
         x_ax, np.insert(np.cumsum(RNDT), 0, 0.0), 'b,:')
ax2.legend(['Proposed (iterative)', 'Proposed', 'Entropy', 'Random'])
ax2.set_xlabel('Number of updates')
ax2.set_ylabel('Cumulative time (seconds, linear scale)')
ax2.set_xticks(np.arange(0, update_cnt+1, 1))
ax2.set_title('Fast methods detail')
ax2.grid(True)

plt.tight_layout()
fig.savefig(resultFolder + 'timeComplexity_5.png', dpi=300)
plt.close(fig)

