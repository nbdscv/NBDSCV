import torch
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def tSNE(net, npc, trainloader):
    # set the model to evaluation mode
    net.eval()

    # tracking variables
    total = 0

    trainFeatures = npc.memory
    trainLabels = torch.LongTensor(trainloader.dataset.targets)

    print("start tSNE...")
    X_embedded = TSNE(n_components=2).fit_transform(trainFeatures.cpu())
    print("done!")

    C = trainLabels.max() + 1
    for c in range(C):
        c_idx = np.where(trainLabels == c)[0]
        plt.scatter(X_embedded[c_idx,0], X_embedded[c_idx,1], label=c)

    plt.legend()
    plt.savefig('tSNE.png')