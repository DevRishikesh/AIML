import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
data = pd.read_csv(&#39;archive.zip&#39;)
data.head()


data.isna().sum()


data.describe()


data.info()

corr = data.corr()
fig = plt.figure(figsize=(10,5))
a = sns.heatmap(corr, cmap='Oranges')
a.set_title("Data Correlation")
