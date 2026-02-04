import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\vboxuser\Desktop\AIML Lab\data.csv")

# Encoding categorical variables
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

# Define features and target
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

# Train the Decision Tree Classifier
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

# Plot the Decision Tree
plt.figure(figsize=(12, 8))  # Adjust figure size for better readability
tree.plot_tree(dtree, 
               feature_names=features, 
               class_names=['NO', 'YES'],  # Class names for the target variable
               filled=True,  # Color the nodes for clarity
               rounded=True,  # Rounded corners for better aesthetics
               fontsize=10)  # Make the text more readable
plt.title("Decision Tree for 'Go' Prediction")
plt.show()


