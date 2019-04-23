import pandas as pd
import numpy as np

train_result = pd.read_csv('train_result.csv')

train_result.plot(x = '0', y = '4')
