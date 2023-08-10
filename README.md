#Training script builders for custom Keras and PyTorch models
This is a desktop application for training fully connected and convolutional neural network models. To do this you need to configure the learning process, specify your model and dataset via the GUI. Then a training script with realtime output of progress will be generated and launched.

#![GUI](/out/mnist/gui.jpg?raw=true)

#Features(Limitations)
- Training, testing and validating models
- Saving model weights and learning history
- Metric and loss function graphs
- Loss functions: MSE, MAE, Categorical Cross-Entropy
- Metrics for testing and validation: MSE, MAE, Accuracy
- Fully connected and CNN models that are inherited from the Keras(tf.karas.Models) or PyTorch(torch.nn.Module) base class.
- Numeric data in a dataset with or without headers
- Optimizer: Adam

