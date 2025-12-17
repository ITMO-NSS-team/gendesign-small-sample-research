import networkx as nx
import random
import math
import numpy as np
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class CustomGraph:
    def __init__(self, in_units: int, out_units: int, new_weights_mode='glorot'):
        self.graph = nx.DiGraph()

        self.in_nodes = range(in_units)
        self.out_nodes = range(in_units, in_units + out_units)

        self.graph.add_nodes_from(self.in_nodes)
        self.graph.add_nodes_from(self.out_nodes)

        self.duplicated_weights_groups = []
        self.duplicated_weights_edges = set()
        self.new_weights_mode = new_weights_mode

    @property
    def parameters_number(self):
        parameters_number = self.graph.number_of_edges()
        for weight_group in self.duplicated_weights_groups:
            parameters_number -= len(weight_group) - 1

        return parameters_number

    @property
    def flops(self):
        return 2 * self.graph.number_of_edges()

    def sample_edges(self, num_edges=None, layer_from: int = None, layer_to: int = None,
                     custom_node_layer_map=None, full_node_layer_map=None, existed_edges=False,
                     max_percent_of_possible_edges=1):
        node_layer_map = self.node_layer_map if not custom_node_layer_map else custom_node_layer_map
        max_layer = max(node_layer_map.values())
        coordinates_map = self.get_coordinates_map(
            custom_node_layer_map=node_layer_map if full_node_layer_map is None else full_node_layer_map
        )

        nodes_layer_from = [node for node, layer in node_layer_map.items() if layer == layer_from]
        nodes_layer_to = [node for node, layer in node_layer_map.items() if layer == layer_to]

        possible_edges = [(u, v) for u in nodes_layer_from for v in nodes_layer_to
                          if (not self.graph.has_edge(u, v) if not existed_edges else self.graph.has_edge(u, v))]
        num_edges = min(num_edges, round(len(possible_edges) * max_percent_of_possible_edges))

        if layer_to == max_layer:
            return random.sample(possible_edges, num_edges)

        possible_edges_distances = [math.sqrt(
            (coordinates_map[u][0] - coordinates_map[v][0]) ** 2 + (coordinates_map[u][1] - coordinates_map[v][1]) ** 2
        ) for u, v in possible_edges]
        sigma = 4 + layer_to * 1.5
        weights = np.array([np.exp(-d / sigma) for d in possible_edges_distances])
        if existed_edges:
            weights = 1 / weights
        weights /= weights.sum()

        indexes = np.random.choice(list(range(len(possible_edges))), size=num_edges, replace=False, p=weights)
        return [possible_edges[i] for i in indexes]

    def add_new_nodes_with_edges(self):
        """Add several new nodes to graph and connects them with 2 existed nodes"""
        edges_num_to_add = random.randint(
            round(self.graph.number_of_edges() * 0.18),
            round(self.graph.number_of_edges() * 0.28)
        )
        nodes_num_to_add = random.randint(
            round(self.graph.number_of_nodes() * 0.12),
            round(self.graph.number_of_nodes() * 0.32)
        )

        existed_nodes_num = max(self.graph.nodes()) + 1
        new_nodes = list(range(existed_nodes_num, existed_nodes_num + nodes_num_to_add))
        nodes_for_new_layer = random.randint(0, 1)

        node_layer_map = self.node_layer_map
        layers = sorted(set(node_layer_map.values()))
        if nodes_for_new_layer == 1:
            layer_for_new_nodes = random.choice(list(range(1, max(layers) + 1)))
            node_layer_map = {node: layer + (1 if layer >= layer_for_new_nodes else 0)
                              for node, layer in node_layer_map.items()}
            for node in new_nodes:
                node_layer_map[node] = layer_for_new_nodes
            full_node_layer_map = node_layer_map
            max_layer = max(layers) + 1
        else:
            layer_for_new_nodes = random.choice(list(range(1, max(layers))))
            full_node_layer_map = deepcopy(node_layer_map)
            node_layer_map = {node: layer
                              for node, layer in node_layer_map.items() if layer != layer_for_new_nodes}
            for node in new_nodes:
                node_layer_map[node] = layer_for_new_nodes
                full_node_layer_map[node] = layer_for_new_nodes
            max_layer = max(layers)

        edges_percents = np.array([0.5, 0.5]) * np.random.uniform(0.5, 1.5, 2)
        edges_percents = [percent / sum(edges_percents) for percent in edges_percents]

        num_edges = round(edges_percents[0] * edges_num_to_add)
        nodes_from = [node for node, layer in node_layer_map.items() if layer == layer_for_new_nodes - 1]
        nodes_to = [node for node, layer in node_layer_map.items() if layer == layer_for_new_nodes]
        initial_edges = []
        for node_to in nodes_to:
            initial_edges.append((random.choice(nodes_from), node_to))
        self.add_new_edges(edges_list=initial_edges)
        self.add_new_edges(edges_list=self.sample_edges(
            num_edges=max(0, num_edges - len(initial_edges)),
            layer_from=layer_for_new_nodes - 1,
            layer_to=layer_for_new_nodes,
            custom_node_layer_map=node_layer_map,
            full_node_layer_map=full_node_layer_map
        ))
        self.add_new_edges(edges_list=self.sample_edges(
            num_edges=round(edges_percents[1] * edges_num_to_add),
            layer_from=layer_for_new_nodes,
            layer_to=layer_for_new_nodes + 1,
            custom_node_layer_map=node_layer_map,
            full_node_layer_map=full_node_layer_map
        ))

        residual_pairs = [(i, layer_for_new_nodes) if i < layer_for_new_nodes else (layer_for_new_nodes, i)
                          for i in range(max_layer + 1) if abs(i - layer_for_new_nodes) > 1]
        residual_pairs = random.sample(
            residual_pairs,
            round(np.random.uniform(0, len(residual_pairs)))
        )
        residual_edges_num = [
            min(
                round(np.random.uniform(
                    edges_num_to_add // len(residual_pairs) // 9,
                    edges_num_to_add // len(residual_pairs) // 4.5
                )),
                4000
            )
            for _ in range(len(residual_pairs))
        ]
        for (layer_from, layer_to), edges_num in zip(residual_pairs, residual_edges_num):
            self.add_new_edges(edges_list=self.sample_edges(
                num_edges=edges_num, layer_from=layer_from, layer_to=layer_to,
                custom_node_layer_map=node_layer_map, full_node_layer_map=full_node_layer_map
            ))

        return [(layer_for_new_nodes - 1, layer_for_new_nodes), (layer_for_new_nodes, layer_for_new_nodes + 1)] + \
            residual_pairs

    def add_new_edges(self, edges_list=None):
        """Add several edges between random nodes"""
        if edges_list is None:
            edges_num_to_add = random.randint(
                round(self.graph.number_of_edges() * 0.2),
                round(self.graph.number_of_edges() * 0.32)
            )
            node_layer_map = self.node_layer_map
            layers = sorted(set(node_layer_map.values()))
            all_layers_pairs = [(i, j) for i in range(max(layers)) for j in range(i + 1, max(layers) + 1)]
            layers_pairs_probabilities = np.array(
                [1 / math.sqrt(pair[1] - pair[0]) for pair in all_layers_pairs]
            )
            noise = np.random.uniform(0.7, 1.3, len(layers_pairs_probabilities))
            layers_pairs_probabilities = layers_pairs_probabilities * noise
            layers_pairs_probabilities /= layers_pairs_probabilities.sum()
            layers_pairs = random.choices(all_layers_pairs, weights=layers_pairs_probabilities, k=3)
            edges_percents = [layers_pairs_probabilities[all_layers_pairs.index(pair)] for pair in layers_pairs]
            edges_percents = [percent / sum(edges_percents) for percent in edges_percents]

            if max(edges_percents) < 0.6 and len(edges_percents) == len(set(edges_percents)):
                max_value = max(edges_percents)
                max_index = edges_percents.index(max_value)
                smaller_values_indexes = [i for i in range(len(edges_percents)) if i != max_index]
                for index in smaller_values_indexes:
                    edges_percents[index] = max(0, edges_percents[index] - (0.6 - max_value) / 2)
                edges_percents[max_index] = 0.6

            sampled_edges = []
            for pair, percent in zip(layers_pairs, edges_percents):
                sampled_edges.extend(self.sample_edges(
                    num_edges=round(percent * edges_num_to_add),
                    layer_from=pair[0],
                    layer_to=pair[1],
                    custom_node_layer_map=node_layer_map
                ))
        else:
            sampled_edges = edges_list

        for edge_to_add in sampled_edges:
            if self.new_weights_mode == 'glorot':
                self.graph.add_edge(*edge_to_add)
            elif self.new_weights_mode == 'preserving':
                self.graph.add_edge(*edge_to_add, weight=1e-3)

        if edges_list is None:
            return layers_pairs

    def __cascade_delete_edge(self, edge_to_remove: tuple):
        self.graph.remove_edge(*edge_to_remove)
        deleted_edges = [edge_to_remove]
        deleted_nodes = set()
        nodes_to_check = [edge_to_remove[0], edge_to_remove[1]]

        while nodes_to_check:
            node = nodes_to_check.pop()
            if node in self.in_nodes or node in self.out_nodes:
                continue

            if not self.graph.in_edges(node):
                nodes_to_check.extend(self.graph.successors(node))
                deleted_edges.extend(self.graph.out_edges(node))
                self.graph.remove_node(node)
                deleted_nodes.add(node)
            elif not self.graph.out_edges(node):
                nodes_to_check.extend(self.graph.predecessors(node))
                deleted_edges.extend(self.graph.in_edges(node))
                self.graph.remove_node(node)
                deleted_nodes.add(node)

        for deleted_edge in deleted_edges:
            if deleted_edge in self.duplicated_weights_edges:
                self.duplicated_weights_edges.discard(deleted_edge)
                for weight_group in self.duplicated_weights_groups:
                    weight_group.discard(deleted_edge)

        return deleted_nodes

    def remove_edges(self):
        """Remove several random edges"""
        edges_num_to_remove = random.randint(
            round(self.graph.number_of_edges() * 0.12),
            round(self.graph.number_of_edges() * 0.22)
        )
        node_layer_map = self.node_layer_map
        layers = sorted(set(node_layer_map.values()))
        need_remove_layer = random.randint(0, 1)
        if need_remove_layer and len(layers) > 3:
            initial_edges = len(self.graph.edges)
            layer_to_remove = random.choice([l for l in layers if l not in [0, max(layers)]])
            pair_edges_map = defaultdict(int)
            for edge in self.graph.edges:
                if layer_to_remove in [node_layer_map[edge[0]], node_layer_map[edge[1]]]:
                    if layer_to_remove == node_layer_map[edge[0]]:
                        layer_from, layer_to = node_layer_map[edge[0]] - 1, node_layer_map[edge[1]]
                    else:
                        layer_from, layer_to = node_layer_map[edge[0]], node_layer_map[edge[1]] + 1
                    pair_edges_map[(layer_from, layer_to)] += 1

            sampled_edges = []
            for (layer_from, layer_to), num_edges in pair_edges_map.items():
                sampled_edges.extend(self.sample_edges(
                    num_edges=round(
                        num_edges *
                        (0.8 if layer_from == layer_to_remove - 1 and layer_to == layer_to_remove + 1 else 0.95)
                    ),
                    layer_from=layer_from,
                    layer_to=layer_to,
                    custom_node_layer_map=node_layer_map
                ))
            self.add_new_edges(edges_list=sampled_edges)
            for node in [n for n in node_layer_map.keys() if node_layer_map[n] == layer_to_remove]:
                self.graph.remove_node(node)
        else:
            all_layers_pairs = [(i, j) for i in range(max(layers)) for j in range(i + 1, max(layers) + 1)]
            pair_edges_map = {pair: 0 for pair in all_layers_pairs}
            for edge in self.graph.edges:
                pair_edges_map[(node_layer_map[edge[0]], node_layer_map[edge[1]])] = \
                    pair_edges_map[(node_layer_map[edge[0]], node_layer_map[edge[1]])] + 1

            all_layers_pairs = [pair for pair in all_layers_pairs if pair_edges_map[pair] > 0]
            layers_pairs_probabilities = np.array(
                [math.sqrt(pair_edges_map[pair]) for pair in all_layers_pairs]
            )
            noise = np.random.uniform(0.8, 1.2, len(layers_pairs_probabilities))
            layers_pairs_probabilities = layers_pairs_probabilities * noise
            layers_pairs_probabilities /= layers_pairs_probabilities.sum()
            layers_pairs = random.choices(
                all_layers_pairs,
                weights=layers_pairs_probabilities,
                k=min(3, len(all_layers_pairs))
            )

            edges_percents = [layers_pairs_probabilities[all_layers_pairs.index(pair)] for pair in layers_pairs]
            edges_percents = [percent / sum(edges_percents) for percent in edges_percents]

            if max(edges_percents) < 0.6 and len(edges_percents) == len(set(edges_percents)) and len(
                    edges_percents) == 3:
                max_value = max(edges_percents)
                max_index = edges_percents.index(max_value)
                smaller_values_indexes = [i for i in range(len(edges_percents)) if i != max_index]
                for index in smaller_values_indexes:
                    edges_percents[index] = max(0, edges_percents[index] - (0.6 - max_value) / 2)
                edges_percents[max_index] = 0.6

            sampled_edges = []
            for pair, percent in zip(layers_pairs, edges_percents):
                sampled_edges.extend(self.sample_edges(
                    num_edges=round(percent * edges_num_to_remove),
                    layer_from=pair[0],
                    layer_to=pair[1],
                    custom_node_layer_map=node_layer_map,
                    existed_edges=True,
                    max_percent_of_possible_edges=(0.5 if pair[0] == pair[1] - 1 else 1)
                ))

            for edge_to_remove in sampled_edges:
                if not self.graph.has_edge(*edge_to_remove):
                    continue
                self.__cascade_delete_edge(edge_to_remove)

    def add_new_weights_duplicates_from_map(self, groups_map: dict):
        """Used to simplify visualization"""
        for group_weight, group_edges in groups_map.items():
            self.duplicated_weights_groups.append(set(group_edges))
            for group_edge in group_edges:
                self.duplicated_weights_edges.add(group_edge)
                if isinstance(group_weight, float):
                    self.graph[group_edge[0]][group_edge[1]]['weight'] = group_weight

    def get_coordinates_map(self, custom_node_layer_map=None):
        node_layer_map = self.node_layer_map if not custom_node_layer_map else custom_node_layer_map
        layer_nodes_map = defaultdict(list)
        for node in sorted(node_layer_map.keys()):
            layer_nodes_map[node_layer_map[node]].append(node)
        coordinates_map = {n: ((n % 1024) // 32, (n % 1024) % 32) for n in
                           self.in_nodes}

        for layer in range(1, max(layer_nodes_map.keys())):
            for node in layer_nodes_map[layer]:
                if not self.graph.has_node(node) or not self.graph.predecessors(node):
                    continue
                node_predecessors = self.graph.predecessors(node)
                node_predecessors_coordinates = [coordinates_map[n] for n in node_predecessors]
                x_coordinates, y_coordinates = \
                    [c[0] for c in node_predecessors_coordinates], [c[1] for c in node_predecessors_coordinates]
                coordinates_map[node] = (
                    sum(x_coordinates) / len(x_coordinates),
                    sum(y_coordinates) / len(y_coordinates)
                )

        return coordinates_map

    @property
    def node_layer_map(self):
        result = {node: 0 for node in self.in_nodes}
        topological_order = list(nx.topological_sort(self.graph))

        for node in topological_order:
            if node not in result:
                continue
            for neighbor in self.graph.successors(node):
                result[neighbor] = max(result.get(neighbor, 0), result[node] + 1)

        last_layer_number = max(1, max(result.values()))
        for node in self.out_nodes:
            result[node] = last_layer_number
        for node in [n for (n, l) in result.items() if l == last_layer_number and n not in self.out_nodes]:
            result.pop(node)

        return result

    def visualize(self, fig_title: str):
        node_layer_map = self.node_layer_map
        max_level = max(node_layer_map.values())

        grouped_edges = set(edge for group in self.duplicated_weights_groups for edge in group)
        default_edges = [edge for edge in self.graph.edges if edge not in grouped_edges]

        colors = list(mcolors.TABLEAU_COLORS.values())
        random.shuffle(colors)

        pos = {}
        layer_spacing = 1.5
        node_spacing = 1.0
        for level in range(max_level + 1):
            nodes_in_level = [node for node, lvl in node_layer_map.items() if lvl == level]
            for i, node in enumerate(nodes_in_level):
                pos[node] = (i * node_spacing, -level * layer_spacing)

        plt.figure(figsize=(10, 6))
        plt.title(fig_title)
        nx.draw_networkx_nodes(self.graph, pos, node_size=200, node_color="lightblue", label="Nodes")

        for idx, edges in enumerate(self.duplicated_weights_groups):
            color = colors[idx % len(colors)]
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color=color, width=0.8)

        nx.draw_networkx_edges(self.graph, pos, edgelist=default_edges, edge_color="black", width=0.8)
        nx.draw_networkx_labels(self.graph, pos, font_size=11, font_color="black")
        plt.show()
