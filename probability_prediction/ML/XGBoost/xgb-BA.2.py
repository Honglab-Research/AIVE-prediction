import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from datetime import datetime as dt
import logging
import time
import os
from xgboost import XGBClassifier
from sklearn.exceptions import NotFittedError


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


file_path = 'filtered_data.txt'

use_clade = ['19A', '19B', '20G', '21K', '20E', '20A', '20C', '20B', '21J', '20D', '20I', '20J', '20H', '21C', '20F', '21L', '21F', '21I', 'recombinant', '21H', '21A', '22C', '21D', '21G', '21E', '21B', '21M', '22D']

data = pd.read_csv(file_path, sep = '\t')

data = data[data['Nextstrain_clade'].isin(use_clade)]
data = data.dropna(subset=['aaSubstitutions'])


classifier = XGBClassifier(tree_method='gpu_hist')  # enabling GPU support

target_labels = ['N440K', 'V445P', 'G446S', 'N460K', 'S477N', 'T478K', 'E484A', 'F486P', 'F490S', 'Q498R', 'N501Y', 'Y505H', 'L452R', 'E484K', 'Q493R', 'G496S', 'F486V', 'K444T', 'N477R', 'N477K', 'N439R', 'Y501R', 'N437R', 'S438R', 'S459R', 'S469R', 'S494R', 'T470R', 'T500R', 'Q493K', 'T470K', 'T500K', 'S469K', 'S494K', 'Y501K', 'N437K', 'N439K', 'N460R', 'S438K', 'S459K']


data['date'] = pd.to_datetime(data['date']).apply(lambda date: date.toordinal())

scaler = StandardScaler()
data[['date']] = scaler.fit_transform(data[['date']])

data['aaSubstitutions'] = data['aaSubstitutions'].str.split(',')

data = data.dropna(subset=['aaSubstitutions'])

data['aaSubstitutions'] = data['aaSubstitutions'].apply(lambda mutations: [mutation for mutation in mutations if mutation in target_labels])


mutations = pd.get_dummies(data['aaSubstitutions'].explode()).sum(level=0)


data = pd.concat([data, mutations], axis=1)

for label in target_labels:
    if label not in data.columns:
        data[label] = 0

data = data.dropna(subset=target_labels)

unused_columns = ['aaSubstitutions']
features = data.drop(target_labels + unused_columns, axis=1)
#features = data.drop(target_labels, axis=1)

features = pd.get_dummies(features, columns=['Nextstrain_clade'])

pipeline = Pipeline(steps=[('classifier', MultiOutputClassifier(classifier))])

X_train, X_test, y_train, y_test = train_test_split(features, data[target_labels], test_size=0.2, random_state=42)


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro') # for multi-label classification
recall = recall_score(y_test, y_pred, average='micro') # for multi-label classification
f1 = f1_score(y_test, y_pred, average='micro') # for multi-label classification

proba_predictions = pipeline.predict_proba(X_test)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 score: {f1}')

# For each target_label
for i, target_label in enumerate(target_labels):
    # Get probabilities for the positive class (mutation is predicted)
    positive_class_proba = proba_predictions[i][:, 1]
    
    # Print mean probability
    print(f"Mean probability for {target_label}: {positive_class_proba.mean()}")

