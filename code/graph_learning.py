import torch
import torch.nn as nn
import numpy as np
import dgl
import dgl.function as fn
import torch.nn.functional as F


class LatentGraph(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        """
        threshold: cutoff for pairwise norm of the embeddings to become an edge
        """
        self.threshold = nn.Threshold(threshold=threshold, value=0)

    def forward(self, input):
        """
        input: pairwise distance of the projected embeddings
        """
        raw_graph = 2 * (1 - torch.sigmoid(input))
        return self.threshold(raw_graph)


# Simple graph convolution with weights
class WeightedGCN(nn.Module):

    def __init__(self, hidden_size):
        super(WeightedGCN, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(hidden_size)

    def forward(self, g):
        with g.local_scope():
            g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'),
                         reduce_func=fn.mean('m', 'h'))
            h = self.linear(g.ndata['h'])
            h = self.bn(h)
            h = F.relu(h)
            h = self.dropout(h)
            g.ndata['h'] = h
            return g


class WeightedGCNModel(nn.Module):
    def __init__(self, hidden_size):
        super(WeightedGCNModel, self).__init__()
        self.conv1 = WeightedGCN(hidden_size)
        self.conv2 = WeightedGCN(hidden_size)

    def forward(self, g):
        g = self.conv1(g)
        g = self.conv2(g)
        return g


# Graph Transformer

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}
    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)
        g.apply_edges(scaled_exp('score', np.sqrt(self.out_dim)))

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.u_mul_e(
            'V_h', 'score', 'V_h'), fn.sum('V_h', 'V_h'))
        g.send_and_recv(eids, fn.u_mul_e(
            'V_h', 'w', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge(
            'score', 'score'), fn.sum('score', 'z'))

    def forward(self, g):

        Q_h = self.Q(g.ndata['h'])
        K_h = self.K(g.ndata['h'])
        V_h = self.V(g.ndata['h'])

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)

        head_out = g.ndata['wV']/g.ndata['z']

        return head_out


class GraphTransformerLayer(nn.Module):
    """
        Param: 
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout=0.0, layer_norm=False, batch_norm=True, residual=True, use_bias=False):
        super(GraphTransformerLayer, self).__init__()

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.attention = MultiHeadAttentionLayer(
            in_dim, out_dim//num_heads, num_heads, use_bias)

        self.lin = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(out_dim)

        # FFN
        self.FFN_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2 = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2 = nn.BatchNorm1d(out_dim)

    def forward(self, g):
        h_in1 = g.ndata['h']  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(g)
        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.lin(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1(h)

        if self.batch_norm:
            h = self.batch_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2(h)

        if self.batch_norm:
            h = self.batch_norm2(h)
        g.ndata['h'] = h
        return g


class GraphTransformerNet(nn.Module):

    def __init__(self, hidden_size):
        super(GraphTransformerNet, self).__init__()

        self.layer_norm = False
        self.batch_norm = True
        self.residual = True

        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_size, hidden_size, 1,
                                                           0.1, self.layer_norm, self.batch_norm, self.residual) for _ in range(0)])

    def forward(self, g):

        # GraphTransformer Layers
        for conv in self.layers:
            g = conv(g)

        return g
