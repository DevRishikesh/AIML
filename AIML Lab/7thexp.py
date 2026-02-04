 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
data = pd.read_csv(r"C:\Users\vboxuser\Desktop\AIML Lab\breast_cancer.csv")
print(data.head())
print(data.isna().sum())
print(data.describe())
print(data.info())
corr = data.corr()
fig = plt.figure(figsize=(10,5))
a = sns.heatmap(corr, cmap='Oranges')
a.set_title("Data Correlation")
plt.show()