import os
import numpy as np
import pandas as pd
#import argparse
import random
import matplotlib.pyplot as pp

oct_data = np.loadtxt("../dataset/lstminput.txt", delimiter='\t')
for i in range(0,35):
    sensor = "Sensor No " + str(i)
    pp.title(sensor)
    pp.plot(oct_data[:,i])
    pp.show()
    print("Variance for sensor {} is {}" .format(i, np.std(oct_data[:,i])))

