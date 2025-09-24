import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import scanpy as sc
import pickle as pkl
import pandas as pd
import numpy as np

data = sc.read_h5ad('/home/iman/scBERT/data/Test_Zheng68K.h5ad')
pred_list = []

with open("/home/iman/scBERT/code/pred_zheng68k_test.txt", "r") as f:
    for line in f:
        pred_list.append(line.strip())

# =========================
# Evaluation metrics
# =========================
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True) 
true_labels = label_dict[label].tolist()
pred_labels = pred_list

# Remove "Unassigned" samples (if needed) for evaluation
true_eval = []
pred_eval = []
for t, p in zip(true_labels, pred_labels):
    if p != "Unassigned":
        true_eval.append(t)
        pred_eval.append(p)

acc = accuracy_score(true_eval, pred_eval)
f1 = f1_score(true_eval, pred_eval, average="weighted")
cm = confusion_matrix(true_eval, pred_eval, labels=label_dict.tolist())

print("Accuracy:", acc)
print("F1 Score:", f1)
print("\nClassification Report:\n", classification_report(true_eval, pred_eval, target_names=label_dict.tolist()))

# =========================
# Confusion Matrix Heatmap
# =========================
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=label_dict.tolist(), yticklabels=label_dict.tolist())
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.savefig("confusion_matrix_Zheng.png", dpi=300)  
plt.close()

# =========================
# t-SNE Plots
# =========================
# فرض: دیتا Zheng68k.h5ad شامل tsne.1 و tsne.2 هست
tsne1 = data.obs["TSNE.1"]
tsne2 = data.obs["TSNE.2"]

# Plot with true labels
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(tsne1, tsne2, c=pd.Categorical(true_labels).codes, cmap="tab20", s=5)
plt.title("t-SNE: True Labels")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")

# Plot with predicted labels
plt.subplot(1,2,2)
plt.scatter(tsne1, tsne2, c=pd.Categorical(pred_labels).codes, cmap="tab20", s=5)
plt.title("t-SNE: Predicted Labels")
plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")

plt.tight_layout()
plt.savefig("tsne_plots_zheng.png", dpi=300)  
plt.close()