from filter import ESKF, Data
import pandas as pd

data = Data()
TrueFilter = ESKF(data)

def run_filter():
    while TrueFilter.iteration < 5000:
        TrueFilter.predict()
        TrueFilter.update()

    #TrueFilter.save_results('quaternion_results.csv') ~~ csv loading only