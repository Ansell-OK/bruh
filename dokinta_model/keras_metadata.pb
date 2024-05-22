
�Groot"_tf_keras_network*�G{"name": "Bidirectional_GRU", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "Bidirectional_GRU", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "TextVectorization", "config": {"name": "text_vectorization", "trainable": true, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "max_tokens": 50000, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 43, "pad_to_max_tokens": false, "sparse": false, "ragged": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "vocabulary_size": 491}, "name": "text_vectorization", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "input_dim": 491, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "name": "embedding", "inbound_nodes": [[["text_vectorization", 0, 0, {}]]]}, {"class_name": "Bidirectional", "config": {"name": "bidirectional", "trainable": true, "dtype": "float32", "layer": {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}}, "merge_mode": "concat"}, "name": "bidirectional", "inbound_nodes": [[["embedding", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["bidirectional", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_4", 0, 0]]}, "shared_object_id": 16, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "input_2"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "input_2"]}, "keras_version": "2.15.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "Bidirectional_GRU", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "TextVectorization", "config": {"name": "text_vectorization", "trainable": true, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "max_tokens": 50000, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 43, "pad_to_max_tokens": false, "sparse": false, "ragged": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "vocabulary_size": 491, "has_input_vocabulary": false}, "name": "text_vectorization", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "input_dim": 491, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "name": "embedding", "inbound_nodes": [[["text_vectorization", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Bidirectional", "config": {"name": "bidirectional", "trainable": true, "dtype": "float32", "layer": {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 8}, "merge_mode": "concat"}, "name": "bidirectional", "inbound_nodes": [[["embedding", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["bidirectional", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 15}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_4", 0, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 18}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>Adam", "config": {"name": "Adam", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": false, "is_legacy_optimizer": false, "learning_rate": 0.0010000000474974513, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_2", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "input_2"}}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "text_vectorization", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "TextVectorization", "config": {"name": "text_vectorization", "trainable": true, "dtype": "string", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "max_tokens": 50000, "standardize": "lower_and_strip_punctuation", "split": "whitespace", "ngrams": null, "output_mode": "int", "output_sequence_length": 43, "pad_to_max_tokens": false, "sparse": false, "ragged": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "vocabulary_size": 491, "has_input_vocabulary": false}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 1, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Embedding", "config": {"name": "embedding", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "input_dim": 491, "output_dim": 128, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": null}, "inbound_nodes": [[["text_vectorization", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 43]}}2
�
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 64, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["bidirectional", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 20}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
�root.layer_with_weights-4"_tf_keras_layer*�{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 4, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
�'root.layer_with_weights-0._lookup_layer"_tf_keras_layer*�{"name": "string_lookup", "trainable": true, "expects_training_arg": false, "dtype": "int64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "int64", "invert": false, "max_tokens": 50000, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "idf_weights": null, "vocabulary": null, "vocabulary_size": 491, "encoding": "utf-8", "has_input_vocabulary": false}, "shared_object_id": 22, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}}2
� 'root.layer_with_weights-2.forward_layer"_tf_keras_rnn_layer*�{"name": "forward_gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "GRU", "config": {"name": "forward_gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 27, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 28}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 43, 128]}}2
�!(root.layer_with_weights-2.backward_layer"_tf_keras_rnn_layer*�{"name": "backward_gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "GRU", "config": {"name": "backward_gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": true, "stateful": false, "unroll": false, "time_major": false, "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 33, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 34}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 43, 128]}}2
�k,root.layer_with_weights-2.forward_layer.cell"_tf_keras_layer*�{"name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 26, "build_input_shape": {"class_name": "__tuple__", "items": [null, 128]}}2
�t-root.layer_with_weights-2.backward_layer.cell"_tf_keras_layer*�{"name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "GRUCell", "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 32, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.8, "recurrent_dropout": 0.8, "implementation": 1, "reset_after": true}, "shared_object_id": 32, "build_input_shape": {"class_name": "__tuple__", "items": [null, 128]}}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 35}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}, "shared_object_id": 18}2