import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import network_functions as nf
import seaborn as sns

# Load the model
model = tf.keras.models.load_model(nf.getPath("../models/80p2n.keras"))

# Load the test data
test_data = train_ds = tf.data.Dataset.load(nf.getPath("../datasets/test"))
test_labels = ["secure", "compromised"]  # Assuming the labels are stored as a .npy file

# Restore the latest checkpoint
latest_checkpoint = nf.getPath("../checkpoints/80p2n.weights.h5")
model.load_weights(latest_checkpoint)

# Make predictions
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# Generate the confusion matrix
conf_matrix = confusion_matrix(test_labels, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()