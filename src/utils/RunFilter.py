from filter import ESKF
import pandas as pd

TrueFilter = ESKF

def run_filter():
    for TrueFilter.iteration < 5000:
        TrueFilter.predict()
        TrueFilter.update()
