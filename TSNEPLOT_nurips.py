import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

adata = sc.read_h5ad("/home/iman/scBERT/data/Test_Neurips.h5ad")
pred_list = []

with open("/home/iman/scBERT/code/pred_Nurips_test.txt", "r") as f:
    for line in f:
        pred_list.append(line.strip())

sc.tl.tsne(adata, n_pcs=50, random_state=42) 

true_labels = adata.obs["cell_type"].tolist()
pred_labels = pred_list

tsne = adata.obsm['X_tsne']
tsne1, tsne2 = tsne[:,0], tsne[:,1]

plt.figure(figsize=(12,5))

# True labels
plt.subplot(1,2,1)
plt.scatter(tsne1, tsne2, c=pd.Categorical(true_labels).codes, cmap="tab20", s=5)
plt.title("t-SNE: True Labels")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")

# Predicted labels
plt.subplot(1,2,2)
plt.scatter(tsne1, tsne2, c=pd.Categorical(pred_labels).codes, cmap="tab20", s=5)
plt.title("t-SNE: Predicted Labels")
plt.xlabel("tSNE-1")
plt.ylabel("tSNE-2")

plt.tight_layout()
plt.savefig("tsne_plots_nurips.png", dpi=300)
plt.close()
