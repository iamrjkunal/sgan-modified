import os
import numpy as np
import pandas as pd
sensor_final = pd.read_csv("sensor_final.csv", delimiter= ",",header =None,low_memory=False)
sensor_final = sensor_final.drop(sensor_final.columns[0], axis=1) 
sensor_final = sensor_final[[4, 3, 1, 2]]
sensor_final = sensor_final.drop(sensor_final.index[0])

oct_data = sensor_final[sensor_final[4]=='10']
oct_data = oct_data.drop(oct_data.columns[0], axis=1) 
oct_data.to_csv('oct_data.csv', sep = '\t', mode = 'w', index=False, header = False)

nov_data = sensor_final[sensor_final[4]=='11']
nov_data = nov_data.drop(nov_data.columns[0], axis=1)
nov_data.to_csv('nov_data.csv', sep = '\t', mode = 'w', index=False, header = False)

dec_data = sensor_final[sensor_final[4]=='12']
dec_data = dec_data.drop(dec_data.columns[0], axis=1)
dec_data.to_csv('dec_data.csv', sep = '\t', mode = 'w', index=False, header = False)

jan_data = sensor_final[sensor_final[4]=='01']
jan_data = jan_data.drop(jan_data.columns[0], axis=1)
jan_data.to_csv('jan_data.csv', sep = '\t', mode = 'w', index=False, header = False)
