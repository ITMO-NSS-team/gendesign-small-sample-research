import tensorflow as tf
from collections import Counter, defaultdict
from typing import Dict, List

from src.graph_utils import CustomGraph


class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, shape: tuple, with_biases: bool, new_layer=True):
        super(CustomDenseLayer, self).__init__()
        self.new_layer = new_layer
        self.dense_shape = shape
        self.sparse_indexes = None
        self.sparse_indexes_map = None
        self.w = None
        self.running_mode = None
        self.with_biases = with_biases

        if self.with_biases:
            self.b = self.add_weight(
                shape=(shape[-1],),
                initializer=tf.constant_initializer(tf.zeros([shape[-1]]).numpy()),
                name='bias'
            )
        else:
            self.b = None
        self.sparsity_mask = None

    def fixate_weights(self, sparse_indexes: List[tuple], flatten_weights: list, b: list):
        assert flatten_weights is None or len(sparse_indexes) == len(flatten_weights)
        if len(sparse_indexes) < self.dense_shape[0] * self.dense_shape[1] * 0.05:
            self.running_mode = 'sparse'
            self.sparse_indexes = tf.constant(sparse_indexes, dtype=tf.int64)
            self.sparse_indexes_map = {tuple(ind): i for i, ind in enumerate(sparse_indexes)}

            limit = tf.sqrt(6.0 / (self.dense_shape[0] + self.dense_shape[1]))
            weights = tf.random.uniform(shape=(len(sparse_indexes),),
                                        minval=-limit,
                                        maxval=limit,
                                        dtype=tf.float32)
            if flatten_weights is not None and [w for w in flatten_weights if w is not None]:
                weights = tf.tensor_scatter_nd_update(
                    weights, [[i] for i in range(len(sparse_indexes)) if flatten_weights[i] is not None],
                    [w for w in flatten_weights if w is not None]
                )
            self.w = self.add_weight(
                shape=(len(sparse_indexes),),
                initializer=tf.constant_initializer(weights.numpy()),
                name='w'
            )
        else:
            self.running_mode = 'masked'
            self.w = self.add_weight(
                shape=self.dense_shape,
                initializer=tf.constant_initializer(tf.keras.initializers.GlorotUniform()(shape=self.dense_shape).numpy()),
                name='w'
            )
            if flatten_weights is not None and [w for w in flatten_weights if w is not None]:
                updated_weights = tf.tensor_scatter_nd_update(
                    self.w,
                    [ind for i, ind in enumerate(sparse_indexes) if flatten_weights[i] is not None],
                    [w for w in flatten_weights if w is not None]
                )
                self.w.assign(updated_weights)

            self.sparsity_mask = tf.zeros(self.dense_shape, dtype=tf.float32)
            self.sparsity_mask = tf.tensor_scatter_nd_update(
                self.sparsity_mask, sparse_indexes, tf.ones(len(sparse_indexes), dtype=tf.float32)
            )

        if b is not None and self.with_biases:
            self.b.assign(b)

    def call(self, inputs):
        if self.running_mode == 'sparse':
            sparse_weights = tf.sparse.SparseTensor(indices=self.sparse_indexes,
                                                    values=self.w,
                                                    dense_shape=self.dense_shape)
            return tf.sparse.sparse_dense_matmul(inputs, sparse_weights) + self.b if self.with_biases else \
                tf.sparse.sparse_dense_matmul(inputs, sparse_weights)
        else:
            masked_w = self.w * self.sparsity_mask
            return tf.matmul(inputs, masked_w) + self.b if self.with_biases else tf.matmul(inputs, masked_w)


def create_model_from_graph(graph: CustomGraph, normalization='layer', without_start_weights=False, new_pairs=None):
    node_layer_map = graph.node_layer_map
    last_layer_number = max(node_layer_map.values())
    inputs = tf.keras.Input(shape=(len(graph.in_nodes),))

    # Create all sparse layers, first without weights initialization.
    # It also can be layers (in fact sparse matrices) that connect, for example, 0 and 2 layers
    layers_connections_pairs = set()
    for node in graph.graph.nodes:
        for neighbor in graph.graph.successors(node):
            if node in node_layer_map and neighbor in node_layer_map:
                layers_connections_pairs.add((node_layer_map[node], node_layer_map[neighbor]))

    layer_neurons_counts = Counter(node_layer_map.values())
    keras_layers = defaultdict(dict)
    for layer_from, layer_to in layers_connections_pairs:
        keras_layers[layer_to][layer_from] = CustomDenseLayer(
            shape=(layer_neurons_counts[layer_from], layer_neurons_counts[layer_to]),
            with_biases=(layer_from + 1 == layer_to),
            new_layer=(layer_from, layer_to) in new_pairs if new_pairs else True
        )

    # Init layers with weights from graph and create mapping between net and graph to save connection between them
    layer_nodes_map = defaultdict(list)
    layer_neurons_map = {}
    for node in sorted(node_layer_map.keys()):
        layer_nodes_map[node_layer_map[node]].append(node)

    for layer in layer_nodes_map.keys():
        layer_nodes_map[layer] = sorted(layer_nodes_map[layer])
        layer_neurons_map[layer] = {node: i for i, node in enumerate(layer_nodes_map[layer])}

    net_graph_weights_mapping = {}
    net_graph_biases_mapping = {}
    duplicated_weights_edges_map = {}
    layers_arrays = {
        layer: {'sparse_indexes': [], 'b': layer.b.numpy() if layer.with_biases else None,
                'flatten_weights': [] if not without_start_weights else None}
        for layer in (v for sub_dict in keras_layers.values() for v in sub_dict.values())
    }
    for u, v, data in graph.graph.edges(data=True):
        if u not in node_layer_map or v not in node_layer_map:
            continue

        u_layer, v_layer = node_layer_map[u], node_layer_map[v]
        u_neuron_number, v_neuron_number = layer_neurons_map[u_layer][u], layer_neurons_map[v_layer][v]
        keras_layer = keras_layers[v_layer][u_layer]
        if 'weight' in data and data['weight'] is not None and not without_start_weights:
            layers_arrays[keras_layer]['flatten_weights'].append(data['weight'])
        elif not without_start_weights:
            layers_arrays[keras_layer]['flatten_weights'].append(None)
        layers_arrays[keras_layer]['sparse_indexes'].append([u_neuron_number, v_neuron_number])
        net_graph_weights_mapping[(keras_layer, u_neuron_number, v_neuron_number)] = (u, v)
        if (u, v) in graph.duplicated_weights_edges:
            duplicated_weights_edges_map[(u, v)] = {
                'layer': keras_layer,
                'neuron_indexes': (u_neuron_number, v_neuron_number)
            }

    for u, data in graph.graph.nodes(data=True):
        if node_layer_map[u] == 0:
            continue

        u_layer = node_layer_map[u]
        u_neuron_number = layer_neurons_map[u_layer][u]
        keras_layer = keras_layers[u_layer][u_layer - 1]

        if 'weight' in data and data['weight'] is not None and keras_layer.with_biases and not without_start_weights:
            layers_arrays[keras_layer]['b'][u_neuron_number] = data['weight']
        net_graph_biases_mapping[(keras_layer, u_neuron_number)] = u

    for keras_layer, np_arrays in layers_arrays.items():
        keras_layer.fixate_weights(
            sparse_indexes=np_arrays['sparse_indexes'], flatten_weights=np_arrays['flatten_weights'], b=np_arrays['b']
        )

    # Calculating outputs
    layers_results = [inputs]
    for current_layer in range(1, last_layer_number + 1):
        activation = tf.keras.activations.softmax if current_layer == last_layer_number else tf.keras.activations.relu
        layer_result = \
            tf.keras.layers.Add()(
                [keras_layers[current_layer][layer_from](layers_results[layer_from])
                 for layer_from in keras_layers[current_layer].keys()]
            )
        if normalization and current_layer != last_layer_number:
            if normalization == 'batch':
                layer_result = tf.keras.layers.BatchNormalization()(layer_result)
            elif normalization == 'layer':
                layer_result = tf.keras.layers.LayerNormalization(center=False, scale=False)(layer_result)

        layer_result = activation(layer_result)
        if current_layer != last_layer_number:
            layer_result = tf.keras.layers.Dropout(0.2)(layer_result)
        layers_results.append(layer_result)

    return tf.keras.Model(inputs=inputs, outputs=layers_results[-1]), \
        duplicated_weights_edges_map, \
        net_graph_weights_mapping, \
        net_graph_biases_mapping


def update_graph_weights(
    graph: CustomGraph,
    net_graph_weights_mapping: Dict[tuple, tuple],
    net_graph_biases_mapping: Dict[tuple, int]
):
    """Updates graph weights after network training to keep them in actual state"""
    layers_np_weights = {layer: layer.w.numpy() for (layer, _, _) in net_graph_weights_mapping.keys()}

    for (layer, ind1, ind2), (node1, node2) in net_graph_weights_mapping.items():
        if layer.running_mode == 'sparse':
            graph.graph[node1][node2]['weight'] = layers_np_weights[layer][layer.sparse_indexes_map[(ind1, ind2)]]
        else:
            graph.graph[node1][node2]['weight'] = layers_np_weights[layer][ind1, ind2]

    for i, weight_group in enumerate(graph.duplicated_weights_groups):
        shared_weight = graph.graph.get_edge_data(*next(iter(weight_group))).get('weight') \
            if len(weight_group) != 0 else None
        for edge in weight_group:
            assert graph.graph.get_edge_data(*edge).get('weight') == shared_weight

    for (layer, ind), node in net_graph_biases_mapping.items():
        graph.graph.nodes[node]['weight'] = layer.b[ind]


def train_model(model, train_dataset, val_dataset, duplicated_weights_edges_map, duplicated_weights_groups, epochs=50,
                lr=5e-4, train_all=False):
    @tf.function
    def train_step(train_images, train_labels):
        with tf.GradientTape() as tape:
            predictions = model(train_images, training=True)
            loss = loss_fn(train_labels, predictions)

        trainable_layers = \
            [layer for layer in model.layers if isinstance(layer, CustomDenseLayer) if layer.new_layer or train_all]
        trainable_weights, trainable_biases = [], []
        for layer in trainable_layers:
            trainable_weights.append(layer.w)
            if layer.with_biases:
                trainable_biases.append(layer.b)

        trainable_variables = trainable_weights + trainable_biases
        grads = tape.gradient(loss, trainable_variables)

        optimizer.apply_gradients(zip(grads, trainable_variables))

        train_loss(loss)
        train_accuracy(train_labels, predictions)

    @tf.function
    def val_step(val_images, val_labels):
        predictions = model(val_images, training=False)
        v_loss = loss_fn(val_labels, predictions)

        val_loss(v_loss)
        val_accuracy(val_labels, predictions)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, weight_decay=1e-3)

    train_loss = tf.keras.metrics.Mean()
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_loss = tf.keras.metrics.Mean()
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_accuracies = []
    best_accuracy = 0
    accuracy_not_updated = 0
    best_model_weights = None
    for epoch in range(epochs):
        accuracy_not_updated += 1
        train_loss.reset_state()
        train_accuracy.reset_state()
        val_loss.reset_state()
        val_accuracy.reset_state()

        for images, labels in train_dataset:
            train_step(images, labels)

        for images, labels in val_dataset:
            val_step(images, labels)

        if val_accuracy.result() > best_accuracy:
            best_accuracy = val_accuracy.result()
            accuracy_not_updated = 0
            best_model_weights = model.get_weights()

        val_accuracies.append(float(val_accuracy.result()))

        if accuracy_not_updated > 18:
            break

        print(
            f"Epoch {epoch + 1}, "
            f"Train loss: {train_loss.result()}, "
            f"Train accuracy: {train_accuracy.result() * 100}, "
            f"Val loss: {val_loss.result()}, "
            f"Val accuracy: {val_accuracy.result() * 100}"
        )

    model.set_weights(best_model_weights)
    top_5_accuracies = sorted(val_accuracies, reverse=True)[:5]
    return sum(top_5_accuracies) / len(top_5_accuracies), best_accuracy
