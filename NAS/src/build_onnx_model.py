import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from collections import Counter, defaultdict


class CustomDenseLayerPytorch(nn.Module):
    def __init__(self, shape: Tuple[int, int], with_biases: bool):
        super(CustomDenseLayerPytorch, self).__init__()
        self.input_dim, self.output_dim = shape
        self.with_biases = with_biases

        self.w = nn.Parameter(torch.zeros(self.input_dim, self.output_dim), requires_grad=False)
        self.register_buffer("sparsity_mask", torch.zeros(self.input_dim, self.output_dim))

        if self.with_biases:
            self.b = nn.Parameter(torch.zeros(self.output_dim), requires_grad=False)
        else:
            self.b = None

    def fixate_weights(self, sparse_indexes: List[Tuple[int, int]], flatten_weights: List[float], b: List[float] = None):
        for (i, j), w in zip(sparse_indexes, flatten_weights):
            if w is not None:
                self.w.data[i, j] = float(w)
                self.sparsity_mask[i, j] = 1.0

        if b is not None and self.with_biases:
            self.b.data = torch.tensor(b, dtype=torch.float32)

    def forward(self, x):
        out = torch.matmul(x, self.w)
        if self.with_biases:
            out = out + self.b
        return out


class CustomGraphModel(nn.Module):
    def __init__(self, graph):
        super(CustomGraphModel, self).__init__()

        self.graph = graph
        self.node_layer_map = graph.node_layer_map
        self.last_layer_number = max(self.node_layer_map.values())

        self.layer_neurons_counts = Counter(self.node_layer_map.values())
        self.layers_connections_pairs = set()
        for node in graph.graph.nodes:
            for neighbor in graph.graph.successors(node):
                if node in self.node_layer_map and neighbor in self.node_layer_map:
                    self.layers_connections_pairs.add((self.node_layer_map[node], self.node_layer_map[neighbor]))

        layer_neurons_counts = Counter(self.node_layer_map.values())
        torch_layers = defaultdict(dict)
        for layer_from, layer_to in self.layers_connections_pairs:
            torch_layers[layer_to][layer_from] = CustomDenseLayerPytorch(
                shape=(layer_neurons_counts[layer_from], layer_neurons_counts[layer_to]),
                with_biases=(layer_from + 1 == layer_to)
            )

        layer_nodes_map = defaultdict(list)
        layer_neurons_map = {}
        for node in sorted(self.node_layer_map.keys()):
            layer_nodes_map[self.node_layer_map[node]].append(node)

        for layer in layer_nodes_map.keys():
            layer_nodes_map[layer] = sorted(layer_nodes_map[layer])
            layer_neurons_map[layer] = {node: i for i, node in enumerate(layer_nodes_map[layer])}

        layers_np_arrays = {
            layer: {'sparse_indexes': [], 'b': layer.b.numpy() if layer.with_biases else None,
                    'flatten_weights': []}
            for layer in (v for sub_dict in torch_layers.values() for v in sub_dict.values())
        }
        for u, v, data in graph.graph.edges(data=True):
            if u not in self.node_layer_map or v not in self.node_layer_map:
                continue

            u_layer, v_layer = self.node_layer_map[u], self.node_layer_map[v]
            u_neuron_number, v_neuron_number = layer_neurons_map[u_layer][u], layer_neurons_map[v_layer][v]
            keras_layer = torch_layers[v_layer][u_layer]
            if 'weight' in data and data['weight'] is not None:
                layers_np_arrays[keras_layer]['flatten_weights'].append(data['weight'])
                layers_np_arrays[keras_layer]['sparse_indexes'].append([u_neuron_number, v_neuron_number])

        for u, data in graph.graph.nodes(data=True):
            if self.node_layer_map[u] == 0:
                continue

            u_layer = self.node_layer_map[u]
            u_neuron_number = layer_neurons_map[u_layer][u]
            keras_layer = torch_layers[u_layer][u_layer - 1]

            if 'weight' in data and data['weight'] is not None and keras_layer.with_biases:
                layers_np_arrays[keras_layer]['b'][u_neuron_number] = data['weight']

        for keras_layer, np_arrays in layers_np_arrays.items():
            keras_layer.fixate_weights(
                sparse_indexes=np_arrays['sparse_indexes'], flatten_weights=np_arrays['flatten_weights'],
                b=np_arrays['b']
            )

        self.torch_layers = torch_layers

    def forward(self, x):
        layer_outputs = {0: x}
        for current_layer in range(1, self.last_layer_number + 1):
            add_inputs = []
            for layer_from in self.torch_layers[current_layer].keys():
                layer = self.torch_layers[current_layer][layer_from]
                add_inputs.append(layer(layer_outputs[int(layer_from)]))

            summed = torch.stack(add_inputs, dim=0).sum(dim=0)
            if current_layer != self.last_layer_number:
                # summed = F.layer_norm(summed, (self.layer_neurons_counts[current_layer],))
                summed = F.relu(summed)
            else:
                summed = F.softmax(summed, dim=-1)
            layer_outputs[current_layer] = summed

        return layer_outputs[self.last_layer_number]


def build_onnx_model(graph, dummy_input, onnx_filename):
    model = CustomGraphModel(graph)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
