import torch
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import test_labels
from late_fusion import fused_preds

# Evaluation Metrics
print("Late Fusion Classification Report:")
print(classification_report(test_labels, fused_preds))

# Confusion Matrix
conf_matrix = confusion_matrix(test_labels, fused_preds)
print("Confusion Matrix:")
print(conf_matrix)
