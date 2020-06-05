import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphStructure(nn.Module):
    def __init__(self, structure, nsamples, low_dim, batch_size, neighbor_size, device):
        super(GraphStructure, self).__init__()
        self.structure = structure
        self.nsamples = nsamples
        self.low_dim = low_dim
        self.batch_size = batch_size
        self.neighbor_size = neighbor_size
        self.device = device

        self.register_buffer('neighbor_indexes_sim', torch.arange(nsamples).view(-1,1))
        self.register_buffer('neighbor_indexes_disim', torch.arange(nsamples).view(-1,1))

    def update(self, npc):
        with torch.no_grad():
            features = npc.memory

            if self.structure == 'BFS':
                tmp_neighbor_indexes_sim = torch.LongTensor(self.nsamples, self.neighbor_size).to(self.device)
                tmp_neighbor_indexes_disim = torch.LongTensor(self.nsamples, self.neighbor_size).to(self.device)
                for start in range(0, self.nsamples, self.batch_size):
                    end = start + self.batch_size
                    end = min(end, self.nsamples)

                    sims = torch.mm(features[start:end], features.t())
                    sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
                    batch_neighbor_sim = sims.topk(self.neighbor_size, largest=True, dim=1)[1]

                    sims = torch.mm(features[start:end], features.t())
                    sims.scatter_(1, self.neighbor_indexes_sim[start:end], 1.)
                    batch_neighbor_disim = sims.topk(self.neighbor_size, largest=False, dim=1)[1]
                    
                    tmp_neighbor_indexes_sim[start:end] = batch_neighbor_sim
                    tmp_neighbor_indexes_disim[start:end] = batch_neighbor_disim

                self.neighbor_indexes_sim = torch.cat([self.neighbor_indexes_sim, tmp_neighbor_indexes_sim], 1)
                self.neighbor_indexes_disim = torch.cat([self.neighbor_indexes_disim, tmp_neighbor_indexes_disim], 1)

            elif self.structure == 'DFS':
                for h in range(1, self.neighbor_size+1):
                    tmp_neighbor_indexes_sim = torch.LongTensor(self.nsamples, 1).to(self.device)
                    tmp_neighbor_indexes_disim = torch.LongTensor(self.nsamples, 1).to(self.device)
                    for start in range(0, self.nsamples, self.batch_size):
                        end = start + self.batch_size
                        end = min(end, self.nsamples)

                        neighbor_indexes_hop_sim = self.neighbor_indexes_sim[start:end, -1]
                        features_hop_sim = features[neighbor_indexes_hop_sim]

                        sims = torch.mm(features_hop_sim, features.t())
                        sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
                        batch_neighbor_sim = sims.topk(1, largest=True, dim=1)[1]
                        
                        sims = torch.mm(features_hop_sim, features.t())
                        sims.scatter_(1, self.neighbor_indexes_sim[start:end], 1.)
                        batch_neighbor_disim = sims.topk(1, largest=False, dim=1)[1]

                        tmp_neighbor_indexes_sim[start:end] = batch_neighbor_sim
                        tmp_neighbor_indexes_disim[start:end] = batch_neighbor_disim

                    self.neighbor_indexes_sim = torch.cat([self.neighbor_indexes_sim, tmp_neighbor_indexes_sim], 1)
                    self.neighbor_indexes_disim = torch.cat([self.neighbor_indexes_disim, tmp_neighbor_indexes_disim], 1)

            elif self.structure == 'random':
                for h in range(1, self.neighbor_size+1):
                    random_hop_idx = torch.randint(h, (self.nsamples, 1)).to(self.device)
                    tmp_neighbor_indexes_sim = torch.LongTensor(self.nsamples, 1).to(self.device)
                    tmp_neighbor_indexes_disim = torch.LongTensor(self.nsamples, 1).to(self.device)
                    for start in range(0, self.nsamples, self.batch_size):
                        end = start + self.batch_size
                        end = min(end, self.nsamples)

                        neighbor_indexes_hop = torch.gather(self.neighbor_indexes_sim[start:end], 1, random_hop_idx[start:end]).view(-1)
                        features_hop = features[neighbor_indexes_hop]
                        
                        sims = torch.mm(features_hop, features.t())
                        sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
                        batch_neighbor_sim = sims.topk(1, largest=True, dim=1)[1]
                        tmp_neighbor_indexes_sim[start:end] = batch_neighbor_sim

                        sims = torch.mm(features_hop, features.t())
                        sims.scatter_(1, self.neighbor_indexes_sim[start:end], 1.)
                        batch_neighbor_disim = sims.topk(1, largest=False, dim=1)[1]
                        tmp_neighbor_indexes_disim[start:end] = batch_neighbor_disim

                    self.neighbor_indexes_sim = torch.cat([self.neighbor_indexes_sim, tmp_neighbor_indexes_sim], 1)
                    self.neighbor_indexes_disim = torch.cat([self.neighbor_indexes_disim, tmp_neighbor_indexes_disim], 1)

            elif self.structure == 'greedy':
                for h in range(1, self.neighbor_size+1):
                    tmp_neighbor_indexes_sim = torch.LongTensor(self.nsamples, 1).to(self.device)
                    tmp_neighbor_indexes_disim = torch.LongTensor(self.nsamples, 1).to(self.device)
                    for start in range(0, self.nsamples, self.batch_size):
                        end = start + self.batch_size
                        end = min(end, self.nsamples)

                        sims_hop = torch.FloatTensor(self.neighbor_indexes_sim[start:end].size(0), 0).to(self.device)
                        for h_inner in range(h):
                            features_hop = features[self.neighbor_indexes_sim[start:end, h_inner]]
                            sims = torch.mm(features_hop, features.t())
                            sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
                            sims_weight = sims.topk(1, largest=True, dim=1)[0]
                            sims_hop = torch.cat([sims_hop, sims_weight], 1)
                        sim_hop_idx = sims_hop.topk(1, largest=True, dim=1)[1]

                        neighbor_indexes_hop = torch.gather(self.neighbor_indexes_sim[start:end], 1, sim_hop_idx).view(-1)
                        features_hop = features[neighbor_indexes_hop]
                        
                        sims = torch.mm(features_hop, features.t())
                        sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
                        batch_neighbor_sim = sims.topk(1, largest=True, dim=1)[1]
                        tmp_neighbor_indexes_sim[start:end] = batch_neighbor_sim

                        sims = torch.mm(features_hop, features.t())
                        sims.scatter_(1, self.neighbor_indexes_sim[start:end], 1.)
                        batch_neighbor_disim = sims.topk(1, largest=False, dim=1)[1]
                        tmp_neighbor_indexes_disim[start:end] = batch_neighbor_disim 

                    self.neighbor_indexes_sim = torch.cat([self.neighbor_indexes_sim, tmp_neighbor_indexes_sim], 1)
                    self.neighbor_indexes_disim = torch.cat([self.neighbor_indexes_disim, tmp_neighbor_indexes_disim], 1)
            print(self.neighbor_indexes_sim.size())
            print('graph construct done')

    # def update(self, npc):
    #     '''
    #     curriculum
    #     '''
    #     with torch.no_grad():
    #         features = npc.memory
    #         hop = self.neighbor_indexes_sim.size(1)

    #         tmp_neighbor_indexes_sim = torch.LongTensor(self.nsamples, self.neighbor_size).to(self.device)
    #         tmp_neighbor_indexes_disim = torch.LongTensor(self.nsamples, self.neighbor_size).to(self.device)

    #         for start in range(0, self.nsamples, self.batch_size):
    #             end = start + self.batch_size
    #             end = min(end, self.nsamples)

    #             if self.structure == 'BFS':
    #                 sims = torch.mm(features[start:end], features.t()) 
    #                 sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
    #                 batch_neighbor_sim = sims.topk(self.neighbor_size, largest=True, dim=1)[1]
    #                 tmp_neighbor_indexes_sim[start:end] = batch_neighbor_sim

    #                 sims = torch.mm(features[start:end], features.t())
    #                 sims.scatter_(1, self.neighbor_indexes_sim[start:end], 1.)
    #                 batch_neighbor_disim = sims.topk(self.neighbor_size, largest=False, dim=1)[1]
    #                 tmp_neighbor_indexes_disim[start:end] = batch_neighbor_disim

    #             elif self.structure == 'DFS':
    #                 neighbor_indexes_hop = self.neighbor_indexes_sim[start:end, -1]
    #                 features_hop = features[neighbor_indexes_hop]
    #                 
    #                 sims = torch.mm(features_hop, features.t())
    #                 sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
    #                 batch_neighbor_sim = sims.topk(self.neighbor_size, largest=True, dim=1)[1]
    #                 tmp_neighbor_indexes_sim[start:end] = batch_neighbor_sim

    #                 sims = torch.mm(features_hop, features.t())
    #                 sims.scatter_(1, self.neighbor_indexes_sim[start:end], 1.)
    #                 batch_neighbor_disim = sims.topk(self.neighbor_size, largest=False, dim=1)[1]
    #                 tmp_neighbor_indexes_disim[start:end] = batch_neighbor_disim

    #             elif self.structure == 'random':
    #                 random_hop_idx = torch.randint(hop, (self.neighbor_indexes[start:end].size(0),1)).to(self.device)
    #                 neighbor_indexes_hop = torch.gather(self.neighbor_indexes[start:end], 1, random_hop_idx).view(-1)
    #                 features_hop = features[neighbor_indexes_hop]
    #                 
    #                 sims = torch.mm(features_hop, features.t())
    #                 sims.scatter_(1, self.neighbor_indexes[start:end], -1.)
    #                 batch_neighbor_sim = sims.topk(self.neighbor_size, largest=True, dim=1)[1]
    #                 tmp_neighbor_indexes_sim[start:end] = batch_neighbor_sim

    #                 sims = torch.mm(features_hop, features.t())
    #                 sims.scatter_(1, self.neighbor_indexes[start:end], 1.)
    #                 batch_neighbor_disim = sims.topk(self.neighbor_size, largest=False, dim=1)[1]
    #                 tmp_neighbor_indexes_disim[start:end] = batch_neighbor_disim

    #             elif self.structure == 'greedy':
    #                 sims_hop = torch.FloatTensor(self.neighbor_indexes_sim[start:end].size(0), 0).to(self.device)
    #                 for h_inner in range(hop):
    #                     features_hop = features[self.neighbor_indexes_sim[start:end, h_inner]]
    #                     sims = torch.mm(features_hop, features.t())
    #                     sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
    #                     sims_weight = sims.topk(1, largest=True, dim=1)[0]
    #                     sims_hop = torch.cat([sims_hop, sims_weight], 1)
    #                 sim_hop_idx = sims_hop.topk(1, largest=True, dim=1)[1]

    #                 neighbor_indexes_hop = torch.gather(self.neighbor_indexes_sim[start:end], 1, sim_hop_idx).view(-1)
    #                 features_hop = features[neighbor_indexes_hop]

    #                 sims = torch.mm(features_hop, features.t())
    #                 sims.scatter_(1, self.neighbor_indexes_sim[start:end], -1.)
    #                 batch_neighbor_sim = sims.topk(1, largest=True, dim=1)[1]
    #                 tmp_neighbor_indexes_sim[start:end] = batch_neighbor_sim

    #                 sims = torch.mm(features_hop, features.t())
    #                 sims.scatter_(1, self.neighbor_indexes_sim[start:end], 1.)
    #                 batch_neighbor_disim = sims.topk(1, largest=False, dim=1)[1]
    #                 tmp_neighbor_indexes_disim[start:end] = batch_neighbor_disim

    #         self.neighbor_indexes_sim = torch.cat([self.neighbor_indexes_sim, tmp_neighbor_indexes_sim], 1) 
    #         self.neighbor_indexes_disim = torch.cat([self.neighbor_indexes_disim, tmp_neighbor_indexes_disim], 1)
    #         print('graph construct done')
