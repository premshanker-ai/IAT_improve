import pickle
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) != 2:
    print("Error - run as `python graphs.py <model_name>`")
    exit()

model_name = sys.argv[1]

history = pickle.load(open(model_name + "/training_history.pickle", "rb"))

kfolds=5
fig, axes = plt.subplots(1, kfolds, figsize=(15, 5))
    
for i in range(0, kfolds):
    axes[i].plot(np.arange(0, 10, 1), history['training']['loss'][i])
    axes[i].plot(np.arange(0, 10, 1), history['validation']['loss'][i])
    
    axes[i].title.set_text("K = %d" % (i + 1))
    
fig.suptitle("Training and Validation Loss")
fig.supxlabel("Epoch")
fig.supylabel("Loss")

plt.tight_layout()
plt.show()
plt.close()

fig, axes = plt.subplots(2, kfolds, figsize=(15, 5))

for i in range(0, kfolds):
    axes[0][i].plot(np.arange(0, 10, 1), history['validation']['PSNR'][i])
    axes[1][i].plot(np.arange(0, 10, 1), history['validation']['SSIM'][i])
    
    axes[0][i].title.set_text("K = %d" % (i + 1))
    axes[1][i].title.set_text("K = %d" % (i + 1))

fig.suptitle("Validation PSNR and SSIM Loss")
fig.supxlabel("Epoch")
fig.supylabel("Score")

plt.tight_layout()
plt.show()
