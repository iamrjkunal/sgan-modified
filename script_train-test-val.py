import os
import numpy as np
import pandas as pd
oct_data = pd.read_csv("datasets/"+ 'oct_data.csv', delimiter= "\t", header =None)
nov_data = pd.read_csv("datasets/"+ 'nov_data.csv', delimiter= "\t", header =None)
dec_data = pd.read_csv("datasets/"+ 'dec_data.csv', delimiter= "\t", header =None)
jan_data = pd.read_csv("datasets/"+ 'jan_data.csv', delimiter= "\t", header =None)

np.savetxt('oct_data.txt', oct_data.values, fmt='%d', delimiter="\t")
np.savetxt('nov_data.txt', nov_data.values, fmt='%d', delimiter="\t")
np.savetxt('dec_data.txt', dec_data.values, fmt='%d', delimiter="\t")
np.savetxt('jan_data.txt', jan_data.values, fmt='%d', delimiter="\t")

oct_data_train = oct_data.iloc[ :200000, :]
oct_data_val = oct_data.iloc[200000:, :]
np.savetxt('oct_data_train.txt', oct_data_train.values, fmt='%d', delimiter="\t")
np.savetxt('oct_data_val.txt', oct_data_val.values, fmt='%d', delimiter="\t")

nov_data_train = nov_data.iloc[ :200000, :]
nov_data_val = nov_data.iloc[200000:, :]
np.savetxt('nov_data_train.txt', nov_data_train.values, fmt='%d', delimiter="\t")
np.savetxt('nov_data_val.txt', nov_data_val.values, fmt='%d', delimiter="\t")

dec_data_train = dec_data.iloc[ :200000, :]
dec_data_val = dec_data.iloc[200000:, :]
np.savetxt('dec_data_train.txt', dec_data_train.values, fmt='%d', delimiter="\t")
np.savetxt('dec_data_val.txt', dec_data_val.values, fmt='%d', delimiter="\t")

jan_data_train = jan_data.iloc[ :20000, :]
jan_data_val = jan_data.iloc[20000:, :]
np.savetxt('jan_data_train.txt', jan_data_train.values, fmt='%d', delimiter="\t")
np.savetxt('jan_data_val.txt', jan_data_val.values, fmt='%d', delimiter="\t")

