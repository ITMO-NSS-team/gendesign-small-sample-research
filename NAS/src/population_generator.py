import pickle
import uuid
import random
import numpy as np
from collections import defaultdict

from src.graph_utils import CustomGraph
from src.model_generator import create_model_from_graph, train_model, update_graph_weights


def generate_population(population, population_num, input_nodes, output_nodes, train_dataset, val_dataset,
                        graphs_folder):
    for individual_index in range(len(population), population_num):
        intermediate_layers_number = individual_index % 4 + 1
        nodes_num = []
        for layers_number in range(intermediate_layers_number):
            nodes_num.append(round(np.random.uniform(
                input_nodes // 2.5 // (layers_number + 1),
                input_nodes * 2 // (layers_number + 1)
            )))

        graph = CustomGraph(
            in_units=input_nodes,
            out_units=output_nodes,
            new_weights_mode='preserving'
        )
        existed_nodes_num = len(graph.graph.nodes())
        layer_nodes_map = defaultdict(list)
        for node in graph.in_nodes:
            layer_nodes_map[0].append(node)

        for node in graph.out_nodes:
            layer_nodes_map[intermediate_layers_number + 1].append(node)

        for layer_num, new_nodes_num in enumerate(nodes_num):
            layer_nodes_map[layer_num + 1].extend(list(range(existed_nodes_num, existed_nodes_num + new_nodes_num)))
            existed_nodes_num += new_nodes_num

        node_layer_map = {}
        for layer, nodes in layer_nodes_map.items():
            for node in nodes:
                node_layer_map[node] = layer

        edges_num = round(np.random.uniform(existed_nodes_num * 5, existed_nodes_num * 20))
        possible_edges_per_layer = np.array(
            [len(layer_nodes_map[layer_num]) * len(layer_nodes_map[layer_num + 1]) *
             (60 if layer_num == intermediate_layers_number else 1)
             for layer_num in range(intermediate_layers_number + 1)]
        )
        noise = np.random.uniform(0.85, 1.15, len(possible_edges_per_layer))
        edges_percents = possible_edges_per_layer * noise
        edges_percents /= edges_percents.sum()
        edges_num_per_layer = edges_percents * edges_num

        for layer_num in range(intermediate_layers_number + 1):
            num_edges = round(edges_num_per_layer[layer_num])
            if layer_num + 1 != max(layer_nodes_map.keys()):
                nodes_from, nodes_to = layer_nodes_map[layer_num], layer_nodes_map[layer_num + 1]
                initial_edges = []
                for node_to in nodes_to:
                    initial_edges.append((random.choice(nodes_from), node_to))
                graph.add_new_edges(edges_list=initial_edges)
            else:
                initial_edges = []

            edges = graph.sample_edges(
                num_edges=max(0, num_edges - len(initial_edges)), layer_from=layer_num, layer_to=layer_num+1,
                custom_node_layer_map=node_layer_map
            )
            if edges:
                graph.add_new_edges(edges_list=edges)

        while([node for node in node_layer_map.keys() if not graph.graph.has_node(node)
               or ((graph.graph.in_degree(node) == 0 or graph.graph.out_degree(node) == 0)
               and node not in graph.in_nodes and node not in graph.out_nodes)]):
            for node in list(node_layer_map.keys()):
                if not graph.graph.has_node(node) or (
                        (graph.graph.in_degree(node) == 0 or graph.graph.out_degree(node) == 0)
                        and node not in graph.in_nodes and node not in graph.out_nodes
                ):
                    node_layer_map.pop(node)
                    if graph.graph.has_node(node):
                        graph.graph.remove_node(node)

        possible_residual_layers = [
            (i, j) for i in range(intermediate_layers_number + 1)
            for j in range(i + 2, intermediate_layers_number + 2)
        ]
        residual_layers = random.sample(
            possible_residual_layers,
            round(np.random.uniform(0, len(possible_residual_layers)))
        )
        residual_edges_num = [
            min(
                round(np.random.uniform(
                    edges_num // len(residual_layers) // 10,
                    edges_num // len(residual_layers) // 5
                )),
                4000
            )
            for _ in range(len(residual_layers))
        ]
        for (layer_from, layer_to), edges_num in zip(residual_layers, residual_edges_num):
            edges = graph.sample_edges(
                num_edges=edges_num, layer_from=layer_from, layer_to=layer_to,
                custom_node_layer_map=node_layer_map
            )
            if edges:
                graph.add_new_edges(edges_list=edges)

        model, duplicated_weights_edges_map, net_graph_weights_mapping, net_graph_biases_mapping = \
            create_model_from_graph(graph, normalization=None, without_start_weights=True)
        avg_val_acc, val_accuracy = train_model(
            model, train_dataset, val_dataset, duplicated_weights_edges_map, graph.duplicated_weights_groups, lr=1e-3,
            epochs=60
        )
        update_graph_weights(graph, net_graph_weights_mapping, net_graph_biases_mapping)
        graph_id = uuid.uuid4()
        population.append({
            'graph_id': graph_id, 'acc': val_accuracy, 'avg_val_acc': avg_val_acc,
            'parameters': graph.parameters_number, 'flops': graph.flops, 'actions': []
        })
        with open(f"{graphs_folder}/{graph_id}.pkl", "wb") as file:
            pickle.dump(graph, file)
        with open(f"{graphs_folder}/0.pkl", "wb") as file:
            pickle.dump(population, file)
