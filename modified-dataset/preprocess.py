import os
import numpy as np
import pandas as pd
data = os.listdir("Dataset")
sensor = [None]*35
for i,content in enumerate(data):
    sensor[i] = pd.read_csv("Dataset/"+ content, delimiter= ",")
    sensor[i]["sensor no."] = content.split(".")[0][5:]
    sensor[i]["month"] = sensor[i]["timestamp"]
    for j in range(0, len(sensor[i]["timestamp"])):
        #if not os.path.exists(sensor[i]["timestamp"][j].split(" ")[0].split("-")[1]):
         #   os.makedirs(sensor[i]["timestamp"][j].split(" ")[0].split("-")[1])
        sensor[i]["month"][j] = sensor[i]["month"][j].split(" ")[0].split("-")[1]
        sensor[i]["timestamp"][j] = sensor[i]["timestamp"][j].split(" ")[0].split("-")[2] + sensor[i]["timestamp"][j].split(" ")[1].split(":")[0] + sensor[i]["timestamp"][j].split(" ")[1].split(":")[1]
sensor_final = pd.concat([i for i in sensor], ignore_index=True)
sensor_final.sort_values(by=['month','timestamp'], inplace=True)
sensor_final.to_csv('sensor_final.csv')

