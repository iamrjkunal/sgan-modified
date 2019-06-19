import os
import numpy as np
import pandas as pd
sensor_final = pd.read_csv("sensor_final.csv", delimiter= ",")
oct_data = sensor_final[sensor_final["month"]=='10']
oct_data.to_csv('oct_data.csv')
nov_data = sensor_final[sensor_final["month"]=='11']
nov_data.to_csv('nov_data.csv')
dec_data = sensor_final[sensor_final["month"]=='12']
dec_data.to_csv('dec_data.csv')
jan_data = sensor_final[sensor_final["month"]=='01']
jan_data.to_csv('jan_data.csv')
