import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Check if dataset exists
data_file = './data.pickle'
if not os.path.exists(data_file):
    print("Error: 'data.pickle' not found. Ensure the dataset is created first.")
    exit()

# Load dataset
with open(data_file, 'rb') as f:
    data_dict = pickle.load(f)

# Ensure uniform data length
max_length = max(len(sample) for sample in data_dict['data'])  # Find max length
data_fixed = [sample + [0] * (max_length - len(sample)) for sample in data_dict['data']]  # Pad shorter samples

data = np.array(data_fixed, dtype=np.float32)  # Convert to NumPy array
labels = np.asarray(data_dict['labels'])

# Split dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict and evaluate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save trained model
model_file = 'model.p'
with open(model_file, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f'Model saved as {model_file}.')
