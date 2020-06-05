import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch import nn
import math


class NonParametricClassifierOP(Function):
    @staticmethod
    def forward(self, x, y, neighbor_indexes, round, structure, memory, params, device):

        T = params[0].item()
        batchSize = x.size(0)

        out = torch.mm(x.data, memory.t())
        out.div_(T) # batchSize * N

        structure = torch.tensor(1) if structure == 'DFS' else torch.tensor(0)
        device = torch.tensor(1) if device == 'cuda' else torch.tensor(0)
        self.save_for_backward(x, y, neighbor_indexes, torch.tensor(round), structure, memory, params, device)
        return out

    @staticmethod
    def backward(self, gradOutput):
        x, y, neighbor_indexes, round, structure, memory, params, device = self.saved_tensors
        structure = 'DFS' if structure == torch.tensor(1) else 'BFS'
        device = 'cuda' if device == torch.tensor(1) else 'cpu'
        # concat the targets for siamese network
        batchSize = gradOutput.size(0)
        T = params[0].item()
        momentum = params[1].item()
        
        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        return gradInput, None, None, None, None, None, None, None, None


class NonParametricClassifier(nn.Module):
    """Non-parametric Classifier
    
    Non-parametric Classifier from
    "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"
    
    Extends:
        nn.Module
    """

    def __init__(self, structure, inputSize, outputSize, T, momentum, device):
        """Non-parametric Classifier initial functin
        
        Initial function for non-parametric classifier
        
        Arguments:
            inputSize {int} -- in-channels dims
            outputSize {int} -- out-channels dims
        
        Keyword Arguments:
            T {int} -- distribution temperate (default: {0.05})
            momentum {int} -- memory update momentum (default: {0.5})
        """
        super(NonParametricClassifier, self).__init__()
        self.structure = structure

        stdv = 1 / math.sqrt(inputSize)
        self.nLem = outputSize

        self.register_buffer('params',
                        torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize)
                                                .mul_(2*stdv).add_(-stdv))
        self.device = device

    def forward(self, x, y, neighbor_indexes, round):
        out = NonParametricClassifierOP.apply(x, y, neighbor_indexes, round, self.structure, self.memory, self.params, self.device)
        return out

    def just_calculate(self, x):
        T = self.params[0].item()
        out = torch.mm(x.data, self.memory.t())
        out.div_(T)
        return out   
