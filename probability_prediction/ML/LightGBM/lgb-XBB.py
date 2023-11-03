import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
import tensorflow as tf
import lightgbm as lgb
from datetime import datetime as dt
import logging
import time
import os
from lightgbm import LGBMClassifier
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)

# Attempt to create a GPU-based LGBMClassifier
try:
    classifier = LGBMClassifier(device='gpu')
    
    # Quick test to see if GPU is being used.
    try:
        classifier.fit(np.array([[1,1],[0,0]]), np.array([0,1]))
    except NotFittedError:
        pass

except Exception as e:
    classifier = LGBMClassifier(device='cpu')
file_path = 'filtered_data.txt'

use_clade = ['21E', '20I', '20J', '21G', '20A', '20C', '20B', '20D', '21K', '19B', '20F', '19A', '21L', '22B', '22D', '22E', '22F', 'recombinant', '20E', '20H', '21J', '21I', '21F', '21A', '21M', '23A', '22C', '22A', '23B', '21D', '21B', '21H', '20G', '21C']

data = pd.read_csv(file_path, sep = '\t')

data = data[data['Nextstrain_clade'].isin(use_clade)]
data = data.dropna(subset=['aaSubstitutions'])


classifier = lgb.LGBMClassifier()

target_labels = ['N440K', 'V445P', 'G446S', 'N460K', 'S477N', 'T478K', 'E484A', 'F486P', 'F490S', 'Q498R', 'N501Y', 'Y505H', 'L452R', 'E484K', 'Q493R', 'G496S', 'F486V', 'K444T', 'N477R', 'N477K', 'N439R', 'Y501R', 'N437R', 'S438R', 'S459R', 'S469R', 'S494R', 'T470R', 'T500R', 'Q493K', 'T470K', 'T500K', 'S469K', 'S494K', 'Y501K', 'N437K', 'N439K', 'N460R', 'S438K', 'S459K']


data['date'] = pd.to_datetime(data['date']).apply(lambda date: date.toordinal())

data.dropna(subset=['date'], inplace=True)

scaler = StandardScaler()
data[['date']] = scaler.fit_transform(data[['date']])

data['aaSubstitutions'] = data['aaSubstitutions'].str.split(',')

data = data.dropna(subset=['aaSubstitutions'])

# Retain only the mutations in the target labels
data['aaSubstitutions'] = data['aaSubstitutions'].apply(lambda mutations: [mutation for mutation in mutations if mutation in target_labels])

# Then convert the lists of mutations to a multi-hot encoded DataFrame
mutations = pd.get_dummies(data['aaSubstitutions'].apply(pd.Series).stack()).sum(level=0)

# Join the mutations data back into the main DataFrame
data = pd.concat([data, mutations], axis=1)

for label in target_labels:
    if label not in data.columns:
        data[label] = 0

data = data.dropna(subset=target_labels)

unused_columns = ['aaSubstitutions']
#print(data.columns)

features = data.drop(target_labels + unused_columns, axis=1)

#features = data.drop(target_labels, axis=1)

features = pd.get_dummies(features, columns=['Nextstrain_clade'])

pipeline = Pipeline(steps=[('classifier', MultiOutputClassifier(classifier))])

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, data[target_labels], test_size=0.2, random_state=42)

start_time = time.time()
pipeline.fit(X_train, y_train)
print('Model trained in %s seconds' % (time.time() - start_time))

y_pred = pipeline.predict(X_test)



# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro') # for multi-label classification
recall = recall_score(y_test, y_pred, average='micro') # for multi-label classification
f1 = f1_score(y_test, y_pred, average='micro') # for multi-label classification


# Print metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 score: {f1}')

# Predict probabilities
proba_predictions = pipeline.predict_proba(X_test)

# For each target_label
for i, target_label in enumerate(target_labels):
    # Get probabilities for the positive class (mutation is predicted)
    positive_class_proba = proba_predictions[i][:, 1]
    
    # Print mean probability
    print(f"Mean probability for {target_label}: {positive_class_proba.mean()}")

