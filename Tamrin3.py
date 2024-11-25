import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import tensorflow as tf
import matplotlib.pyplot as plt

# Load dataset
dataset_path = "C:\\Users\\Asus\\Desktop\\IOT\\Tamrin 3\\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
data = pd.read_csv(dataset_path)

# Rename columns
data.columns = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'
]

# Preprocess data
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Separate features and labels
features = data.drop('Label', axis=1)
labels = LabelEncoder().fit_transform(data['Label'])

# Standardize features
features = StandardScaler().fit_transform(features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Create time series data
def create_sequences(data, labels, timesteps=100):
    sequences, sequence_labels = [], []
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i+timesteps])
        sequence_labels.append(labels[i+timesteps-1])
    return np.array(sequences), np.array(sequence_labels)

X_train_ts, y_train_ts = create_sequences(X_train, y_train)
X_test_ts, y_test_ts = create_sequences(X_test, y_test)

# Define and train LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train_ts.shape[1], X_train_ts.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_ts, y_train_ts, epochs=10, batch_size=64, validation_data=(X_test_ts, y_test_ts))

# Evaluate model
y_pred = (model.predict(X_test_ts) > 0.5).astype("int32")
print(f"Accuracy: {accuracy_score(y_test_ts, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test_ts, y_pred))
print(f"ROC AUC Score: {roc_auc_score(y_test_ts, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test_ts, y_pred):.4f}")

# Plot accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.show()
