import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 10, 5.8

threshold, score, std = np.loadtxt("feature_scores.csv", unpack=True,
                                   delimiter=",")

plt.errorbar(threshold, score, yerr=std, fmt=".")
plt.xlabel("Feature Score Threshold")
plt.ylabel("CV Score")
plt.ylim(0.113, 0.135)
plt.grid()
# plt.show()
plt.savefig("feature_scores.pdf")
