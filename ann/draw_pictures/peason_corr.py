import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../../dataset/train/train.csv')
# calculate the correlation matrix
corr = df.corr()
print(corr)
