# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve)

# Load Processed Dataset
df = pd.read_csv("bank-full-processed.csv", sep=";")

# Separate Features and Target
X = df.drop(columns=["y"])
y = df["y"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Model
model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_score)
print("MLP Neural Network Results")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))
print("Log Loss :", round(log_loss(y_test, y_score), 4))
# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Values for Graphs
cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_score)
prec_values, rec_values, _ = precision_recall_curve(y_test, y_score)

# Evaluation Graphs
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Graph 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["no", "yes"], yticklabels=["no", "yes"], ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("Actual")

# Graph 2: ROC Curve
axes[0, 1].plot(fpr, tpr, label="AUC = " + str(round(roc_auc, 3)))
axes[0, 1].plot([0, 1], [0, 1], "k--", label="Random Guess")
axes[0, 1].set_title("ROC Curve")
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].legend()

# Graph 3: Precision-Recall Curve
axes[1, 0].plot(rec_values, prec_values)
axes[1, 0].set_title("Precision-Recall Curve")
axes[1, 0].set_xlabel("Recall")
axes[1, 0].set_ylabel("Precision")

# Graph 4: Metrics Bar Chart
metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
metric_values = [accuracy, precision, recall, f1, roc_auc]
axes[1, 1].bar(metric_names, metric_values, color="steelblue")
axes[1, 1].set_ylim(0, 1.1)
axes[1, 1].set_title("Performance Metrics")
for i, v in enumerate(metric_values):
    axes[1, 1].text(i, v + 0.02, str(round(v, 2)), ha="center")

plt.suptitle("MLP Neural Network - Evaluation Graphs", fontsize=14)
plt.tight_layout()
plt.show()

# Graph 5: Training Loss and Validation Accuracy
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.subplot(1, 2, 2)
plt.plot(model.validation_scores_)
plt.title("Validation Accuracy Curve")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.suptitle("MLP Neural Network - Training Graphs")
plt.tight_layout()
plt.show()