
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

# Load dataset
heartDisease = pd.read_csv(r'C:\Users\vboxuser\Desktop\AIML Lab\heart.csv')
heartDisease = heartDisease.replace('?', np.nan)
heartDisease = heartDisease.dropna()

print('Sample instances from the dataset are given below')
print(heartDisease.head())

print('\nAttributes and datatypes')
print(heartDisease.dtypes)

# Define model
model = DiscreteBayesianNetwork([
    ('age', 'heartdisease'),
    ('sex', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'chol')
])

print('\nLearning CPD using Maximum Likelihood Estimator')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

print('\nInferencing with Bayesian Network:')
infer = VariableElimination(model)

print('\n1. Probability of HeartDisease given evidence = restecg')
q1 = infer.query(variables=['heartdisease'], evidence={'restecg': 1})
print(q1)

print('\n2. Probability of HeartDisease given evidence = cp')
q2 = infer.query(variables=['heartdisease'], evidence={'cp': 2})
print(q2)