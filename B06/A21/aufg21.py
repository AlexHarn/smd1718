import numpy as np
import pandas as pd

def infgain(data, target, cut):
    p_ytrue = np.count_nonzero(target == True)
    p_yfalse = (len(target) - p_ytrue)/len(target)
    p_ytrue /= len(target)
    H_y = -p_ytrue*np.log2(p_ytrue) - p_yfalse*np.log2(p_yfalse)


if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    # print(data)
    print(data["Temperatur"])
