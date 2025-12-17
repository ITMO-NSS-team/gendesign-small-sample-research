from typing import List
from itertools import product
from collections import defaultdict
import math
import numpy as np

from src.graph_utils import CustomGraph


def create_graph(in_shape: tuple, out_nodes: int, layers_config: List[dict]):
    in_units = math.prod(in_shape)
    graph = CustomGraph(in_units=in_units, out_units=out_nodes)
    current_nodes = np.array([i for i in range(in_units)]).reshape(in_shape)

    for layer_config in layers_config:
        if layer_config['type'] == 'conv':
            existed_nodes_num = len(graph.graph.nodes())

            new_nodes = np.array(
                [
                    [
                        [
                            existed_nodes_num + c * current_nodes.shape[0] * current_nodes.shape[1] + w +
                            h * current_nodes.shape[1]
                            for c in range(layer_config['out_channels'])
                        ]
                        for w in range(current_nodes.shape[1])
                    ]
                    for h in range(current_nodes.shape[0])
                ]
            )

            weight_groups = defaultdict(list)
            new_edges = set()

            for h, w, in_c, out_c in product(
                *[range(dim) for dim in current_nodes.shape + (layer_config['out_channels'], )]
            ):
                for kernel_h in range(-(layer_config['kernel'] // 2), layer_config['kernel'] // 2 + 1):
                    for kernel_w in range(-(layer_config['kernel'] // 2), layer_config['kernel'] // 2 + 1):
                        if 0 <= h + kernel_h < current_nodes.shape[0] and \
                                0 <= w + kernel_w < current_nodes.shape[1]:
                            edge = (
                                current_nodes[h][w][in_c],
                                new_nodes[h + kernel_h][w + kernel_w][out_c]
                            )
                            if edge not in new_edges:
                                new_edges.add(edge)
                            else:
                                print(f'edge - {edge}, {h}, {kernel_h}, {w}, {kernel_w}, {in_c}, {out_c}')
                            weight_groups[(kernel_h, kernel_w, in_c, out_c)].append(edge)

            graph.add_new_edges(edges_list=new_edges)
            # graph.add_new_weights_duplicates_from_map(weight_groups)

            current_nodes = new_nodes

        elif layer_config['type'] == 'pooling':
            current_node_num = len(graph.graph.nodes())
            new_nodes = []
            kernel = layer_config['kernel']
            stride = layer_config['stride']
            new_edges = set()

            for channel_num in range(current_nodes.shape[2]):
                new_nodes.append([])
                left_upper_h, left_upper_w = 0, 0
                current_channel_list = new_nodes[channel_num]
                current_channel_list.append([])
                current_h_list = current_channel_list[0]
                while True:
                    current_h_list.append(current_node_num)
                    for kernel_h, kernel_w in product(*[range(kernel), range(kernel)]):
                        if left_upper_h + kernel_h < current_nodes.shape[0] and \
                                left_upper_w + kernel_w < current_nodes.shape[1]:
                            new_edges.add((
                                current_nodes[left_upper_h + kernel_h][left_upper_w + kernel_w][channel_num],
                                current_node_num
                            ))

                    current_node_num += 1
                    left_upper_w += stride
                    if left_upper_w >= current_nodes.shape[1]:
                        left_upper_w = 0
                        left_upper_h += stride
                        if left_upper_h >= current_nodes.shape[0]:
                            break
                        current_channel_list.append([])
                        current_h_list = current_channel_list[-1]

            graph.add_new_edges(edges_list=new_edges)
            current_nodes = np.array(new_nodes)
            # set channel last
            current_nodes = np.transpose(current_nodes, (1, 2, 0))

        elif layer_config['type'] == 'fully_connected':
            if current_nodes.ndim != 1:
                current_nodes = current_nodes.flatten()

            if layer_config == layers_config[-1]:
                new_nodes = np.array(graph.out_nodes)
            else:
                existed_nodes_num = len(graph.graph.nodes())
                new_nodes = np.array(list(range(existed_nodes_num, existed_nodes_num + layer_config['out_units'])))

            new_edges = []
            for current_node, new_node in product(*[current_nodes, new_nodes]):
                new_edges.append((current_node, new_node))

            graph.add_new_edges(edges_list=new_edges)

            current_nodes = new_nodes

    return graph
