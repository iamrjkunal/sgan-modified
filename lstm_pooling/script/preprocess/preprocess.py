import os
import numpy as np
import pandas as pd
data = pd.read_csv('../../dataset/oct_data.txt', delimiter='\t')



 
ans = np.array([data.iloc[ 0:35, 2].values])
for i in range(1,int(len(data)/35)):
    start = 35*i
    end = start + 35
    ans = np.append(ans, np.array([data.iloc[ start:end, 2].values]), axis =0)
#np.savetxt('lstminput.txt', ans, delimiter = "\t", fmt='%d')  
