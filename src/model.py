import torch
from torch import nn
import math
import torch.nn.functional as F
import torch.optim as optim

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            m.bias.data.fill_(0.01)
# Dense Hypergraph Neural Network Convolution layer
class DenseHGNNConv(nn.Module):
    def __init__(self, self_in_channels, out_channels):
        super(DenseHGNNConv, self).__init__()
        self.theta = nn.Linear(self_in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta.weight)
        nn.init.zeros_(self.theta.bias)
    # Compute degree matrices: vertex degree D_v and hyperedge degree D_e with inverse normalization
    def forward(self, x, dynamic_H):
        # x: [B, N, In], dynamic_H: [B, N, M]
        x = self.theta(x) 

        d_v = dynamic_H.sum(dim=2, keepdim=True).clamp(min=1.0)
        d_v_inv = torch.pow(d_v, -0.5)

        d_e = dynamic_H.sum(dim=1)
        d_e = d_e.clamp(min=1.0)
        d_e_inv = torch.pow(d_e, -1.0).unsqueeze(-1)
        # Message passing: node -> hyperedge -> node, followed by a residual connection
        x_norm = x * d_v_inv
        H_t = dynamic_H.permute(0, 2, 1)
        hyperedge_attr = torch.matmul(H_t, x_norm)

        hyperedge_attr = hyperedge_attr * d_e_inv
        out = torch.matmul(dynamic_H, hyperedge_attr)
        out = out * d_v_inv
        out = out + x 

        return out
class CSGNet(nn.Module):
    def __init__(self, in_channel, mid_channel=16, out_channel=1, num_nodes=2000, global_adj=None, **args):
        super(CSGNet, self).__init__()


        if global_adj is not None:
            self.register_buffer('global_H', global_adj)
        else:
            self.register_buffer('global_H', torch.zeros(num_nodes, 200))

        self.mid_channel = mid_channel
        self.num_nodes = num_nodes
        self.dropout_ratio = args.get('dropout_ratio', 0.2)
        
        self.conv1 = DenseHGNNConv(in_channel, mid_channel)
        self.bn1 = nn.LayerNorm([num_nodes, mid_channel])
        self.act1 = nn.ReLU()

        self.global_conv1_dim = args.get('global_conv1_dim', 12)
        self.global_conv2_dim = args.get('global_conv2_dim', 4)

        self.global_conv1 = nn.Conv2d(mid_channel, self.global_conv1_dim, [1, 1])
        self.global_bn1 = nn.BatchNorm2d(self.global_conv1_dim)
        self.global_act1 = nn.ReLU()

        self.global_conv2 = nn.Conv2d(self.global_conv1_dim, self.global_conv2_dim, [1, 1])
        self.global_bn2 = nn.BatchNorm2d(self.global_conv2_dim)
        self.global_act2 = nn.ReLU()

        last_feature_node = args.get('last_feature_node', 64)
        input_dim = self.global_conv2_dim * num_nodes
        
        channel_list = [input_dim, 256, last_feature_node]
        self.nn = []
        for idx in range(len(channel_list) - 1):
            self.nn.append(nn.Linear(channel_list[idx], channel_list[idx+1]))
            self.nn.append(nn.BatchNorm1d(channel_list[idx+1]))
            if self.dropout_ratio > 0:
                self.nn.append(nn.Dropout(self.dropout_ratio))
            self.nn.append(nn.ReLU())
        
        self.global_fc_nn = nn.Sequential(*self.nn)
        self.fc1 = nn.Linear(last_feature_node, out_channel)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        nn.init.kaiming_normal_(self.global_conv1.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.global_conv2.weight, mode='fan_out')
        self.global_fc_nn.apply(init_weights)
        self.fc1.apply(init_weights)

    def forward(self, data, get_latent_variable=False):
        x = data[0] if isinstance(data, list) else data
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        global_H_batch = self.global_H.unsqueeze(0).repeat(x.size(0), 1, 1)
        # Gating global hypergraph skeleton with individual cell gene expression to construct cell-specific hypergraph
        gene_activation = torch.tanh(x.squeeze(-1)).unsqueeze(-1)
        dynamic_H = global_H_batch * gene_activation

        x = self.conv1(x, dynamic_H)
        self.current_dynamic_H = dynamic_H

        x = self.act1(x)
        x = self.bn1(x)
        
        x = x.permute(0, 2, 1).unsqueeze(-1)
        
        x = self.global_act1(self.global_bn1(self.global_conv1(x)))
        x = self.global_act2(self.global_bn2(self.global_conv2(x)))
        
        x = x.squeeze(-1)
        num_samples = x.size(0)
        x = x.reshape(num_samples, -1)
        
        latent = self.global_fc_nn(x)
        out = self.fc1(latent)

        if get_latent_variable:
            return latent, out
        else:
            return out

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, dim=-1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = - mean_log_prob_pos
        return loss.mean()