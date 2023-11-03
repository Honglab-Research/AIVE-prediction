import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

file_path = 'filtered_data.txt'

use_clade = ['19A', '19B', '20G', '20E', '20A', '20C', '20B', '21J', '20D', '20I', '20J', '20H', '21C', '20F', '21F', '21I', '21H', '21A', '21D', '21G', '21E', '21B']

data = pd.read_csv(file_path, sep = '\t')

data = data[data['Nextstrain_clade'].isin(use_clade)]

data = data.dropna(subset=['aaSubstitutions'])


target_labels = ['N440K', 'G446S', 'N460K', 'S477N', 'T478K', 'E484A', 'F490S', 'Q498R', 'N501Y', 'Y505H', 'L452R', 'E484K', 'Q493R', 'G496S', 'F486V', 'K444T', 'S494R', 'Q493K', 'T470K', 'N439K']

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

features = data.drop(target_labels + unused_columns, axis=1)

features = pd.get_dummies(features, columns=['Nextstrain_clade'])

pipeline = Pipeline(steps=[('classifier', MultiOutputClassifier(RandomForestClassifier()))])

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, data[target_labels], test_size=0.2, random_state=42)

# Create dictionaries to store the results
train_accuracy_results = {}
valid_accuracy_results = {}
mean_probability_results = {}

# Iterate over all target_labels
for i, target_label in enumerate(target_labels):
    print(f"Training model for {target_label}")

    # Initialize the classifier
    classifier = RandomForestClassifier()
    
    # Fit the model
    classifier.fit(X_train, y_train[target_label])
    
    # Make predictions
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train[target_label], y_train_pred)
    valid_accuracy = accuracy_score(y_test[target_label], y_test_pred)
    
    # Calculate mean probability
    y_test_prob = classifier.predict_proba(X_test)[:, 1] # Get the probabilities for the positive outcome only
    mean_probability = np.mean(y_test_prob) # Get the mean probability
    
    # Print metrics
    print(f'Train Accuracy: {train_accuracy}')
    print(f'Valid Accuracy: {valid_accuracy}')
    print(f'Mean Probability: {mean_probability}')
    
    # Store accuracies in the dictionaries
    train_accuracy_results[target_label] = train_accuracy
    valid_accuracy_results[target_label] = valid_accuracy
    mean_probability_results[target_label] = mean_probability
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(target_labels[:i+1])), list(train_accuracy_results.values()), label="Train Accuracy")
    plt.plot(np.arange(len(target_labels[:i+1])), list(valid_accuracy_results.values()), label="Valid Accuracy")
    plt.plot(np.arange(len(target_labels[:i+1])), list(mean_probability_results.values()), label="Mean Probability")
    plt.xlabel("Iteration")
    plt.ylabel("Metric Value")
    plt.title(f"Metrics for {target_label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/data/cov_metadata/rf-before/rf-{target_label}.png', dpi=300)

