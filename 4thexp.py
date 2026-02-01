import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

# Load dataset
heartDisease = pd.read_csv('heart.csv')

# Replace missing values and drop them
heartDisease = heartDisease.replace('?', np.nan)
heartDisease = heartDisease.dropna()

print('Sample instances from the dataset are given below')
print(heartDisease.head())

print('\nAttributes and datatypes')
print(heartDisease.dtypes)

# Define Bayesian Model
model = BayesianModel([
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
HeartDiseasetest_infer = VariableElimination(model)

print('\n1. Probability of HeartDisease given evidence = restecg')
q1 = HeartDiseasetest_infer.query(
    variables=['heartdisease'],
    evidence={'restecg': 1}
)
print(q1)

print('\n2. Probability of HeartDisease given evidence = cp')
q2 = HeartDiseasetest_infer.query(
    variables=['heartdisease'],
    evidence={'cp': 2}
)
print(q2)
