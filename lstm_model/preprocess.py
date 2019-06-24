import os
import numpy as np
import pandas as pd
oct_data = pd.read_csv('oct_data.txt', delimiter= "\t", header =None)




 
ans = np.array([oct_data.iloc[ 0:35, 2].values])
for i in range(1,int(len(oct_data)/35)):
    start = 35*i
    end = start + 35
    ans = np.append(ans, np.array([oct_data.iloc[ start:end, 2].values]), axis =0)
np.savetxt('lstminput.txt', ans, fmt='%d')  
