import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

file_path = 'filtered_data.txt'

use_clade = ['19A', '19B', '20G', '21K', '20E', '20A', '20C', '20B', '21J', '20D', '20I', '20J', '20H', '21C', '20F', '22B', '21L', '21F', '21I', 'recombinant', '21H', '21A', '22C', '22A', '21D', '21G', '22E', '21E', '21B', '21M', '22D']

data = pd.read_csv(file_path, sep = '\t')

data = data[data['Nextstrain_clade'].isin(use_clade)]
data = data.dropna(subset=['aaSubstitutions'])


target_labels = ['N440K', 'V445P', 'G446S', 'N460K', 'S477N', 'T478K', 'E484A', 'F486P', 'F490S', 'Q498R', 'N501Y', 'Y505H', 'L452R', 'E484K', 'Q493R', 'G496S', 'F486V', 'K444T', 'S494R', 'T500R', 'Q493K', 'T470K', 'N439K', 'N460R']


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
features = pd.get_dummies(features, columns=['Nextstrain_clade'])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(features, data[target_labels], test_size=0.2, random_state=42)

# build the model
inputs = Input(shape=(X_train.shape[1],))
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
outputs = [Dense(1, activation='sigmoid')(x) for _ in target_labels]

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(X_train, [y_train[label] for label in target_labels], epochs=10, batch_size=2**15, validation_split=0.2)

# prediction
proba_predictions = model.predict(X_test)

# For each target_label
for i, target_label in enumerate(target_labels):
    # Get probabilities for the positive class (mutation is predicted)
    positive_class_proba = proba_predictions[i]
    
    # Print mean probability
    print(f"Mean probability for {target_label}: {positive_class_proba.mean()}")

y_pred = np.round(np.concatenate(proba_predictions, axis=1))  # rounding to get 0 or 1

# evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 score: {f1}')




