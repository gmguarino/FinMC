import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Parsing plotting arguments")

parser.add_argument('--path',
                    type=str,
                    help='the path to the data to be plotted',
                    default="./data.dat")

parser.add_argument('--save',
                    type=str,
                    help='filepath where to save the image',
                    default="./plot.png")


def smooth(array, kernel_size=7):
    kernel = np.ones(kernel_size)/kernel_size
    smooth = np.convolve(array, kernel, mode='same')
    return smooth

args = parser.parse_args()
with open(args.path, "r") as f:
    string = f.read()

plt.figure(figsize=(8, 4))
array = string.split(", ")[:-1]
array = np.array(list(map(float, array)))
values, bins, _ = plt.hist(array[array > 0], bins=100, density=True, color="green", alpha=0.5)
plt.plot((bins[:-1] + bins[1:]) / 2, smooth(values), color='red', lw=3)
plt.ylim([0, values.max() * 2])
plt.xlabel("Simulated Values $")
plt.ylabel("Density Count")
plt.title("Non-Zero Simulation Result of European Barrier Options")
plt.savefig(args.save)
