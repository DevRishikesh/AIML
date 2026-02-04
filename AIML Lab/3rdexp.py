import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Dataset
Outlook = [
    'sunny', 'sunny', 'sunny', 'sunny', 'overcast',
    'rainy', 'rainy', 'overcast', 'sunny', 'sunny',
    'rainy', 'sunny', 'overcast', 'overcast', 'rainy'
]

Temp = [
    'hot', 'hot', 'hot', 'mild', 'cool',
    'cool', 'cool', 'mild', 'cool', 'mild',
    'mild', 'mild', 'hot', 'mild', 'mild'
]

Humidity = [
    'high', 'high', 'high', 'high', 'normal',
    'normal', 'normal', 'high', 'normal', 'normal',
    'normal', 'high', 'normal', 'high', 'high'
]

Windy = [
    'false', 'true', 'false', 'true', 'false',
    'false', 'false', 'true', 'false', 'false',
    'true', 'true', 'false', 'true', 'false'
]

Play = [
    'no', 'no', 'yes', 'yes', 'yes',
    'no', 'yes', 'no', 'yes', 'yes',
    'yes', 'yes', 'yes', 'no', 'yes'
]

# Create DataFrame
weatherdata = pd.DataFrame({
    'Outlook': Outlook,
    'Temp': Temp,
    'Humidity': Humidity,
    'Windy': Windy,
    'Play': Play
})

print(weatherdata.head())

# Label Encoding
le = preprocessing.LabelEncoder()

outlook = le.fit_transform(Outlook)
temp = le.fit_transform(Temp)
humidity = le.fit_transform(Humidity)
windy = le.fit_transform(Windy)
play = le.fit_transform(Play)

# Encoded features DataFrame
weatherFeatures = pd.DataFrame({
    'outlook': outlook,
    'temp': temp,
    'humidity': humidity,
    'windy': windy
})

print(weatherFeatures.head())
print("Play =", play)

# Scatter plot (Outlook vs Windy)
data2d = weatherFeatures.loc[:, ['outlook', 'windy']]

pos = data2d.loc[play == 1]
neg = data2d.loc[play == 0]

plt.scatter(pos.iloc[:, 0], pos.iloc[:, 1], label='Play')
plt.scatter(neg.iloc[:, 0], neg.iloc[:, 1], label='Not Play')

plt.xlabel('Outlook')
plt.ylabel('Windy')
plt.title('Weather Data')
plt.legend()
plt.show()
