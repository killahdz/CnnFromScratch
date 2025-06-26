CNN from Scratch
Overview
This project is a C# implementation of a Convolutional Neural Network (CNN) built from the ground up for educational purposes. It aims to help developers and learners understand the inner workings of CNNs by providing a clear, modular, and extensible codebase. The project includes a complete training pipeline for the CIFAR-10 dataset, supporting both simple and VGG11-inspired architectures, with options for SGD and Adam optimizers.
The implementation covers key components of a CNN, including convolutional layers, batch normalization, ReLU activation, max pooling, dropout, and dense layers, along with training and evaluation logic. It is designed to be readable and well-documented, making it a valuable resource for those studying deep learning concepts.
Features

Custom CNN Implementation: Includes core layers (Convolution, BatchNorm, ReLU, MaxPool, Dropout, Dense) built from scratch.
CIFAR-10 Dataset Support: Automatically downloads and processes the CIFAR-10 dataset for training and evaluation.
Flexible Model Architectures:
Simple CNN: 3x(Conv-BatchNorm-ReLU-Conv-BatchNorm-ReLU-MaxPool) → Dense → ReLU → Dense → Softmax.
VGG11-inspired CNN for more complex tasks.


Optimizer Options:
Stochastic Gradient Descent (SGD) with momentum and gradient clipping.
Adam optimizer for adaptive learning.


Hyperparameter Tuning: Configurable learning rate, batch size, dropout rate, and epochs via console prompts.
Training Pipeline: Supports batch processing, learning rate decay, and validation accuracy tracking.
Model Serialization: Save trained models to JSON for reuse.
Error Handling: Robust checks for data loading and model configuration.

Purpose
This project was developed to deepen the understanding of CNNs by implementing them without relying on high-level frameworks like TensorFlow or PyTorch. It serves as an educational tool for students, developers, and enthusiasts who want to learn how CNNs work under the hood, including forward/backward passes, gradient computation, and optimization.
Getting Started
Prerequisites

.NET SDK (version 6.0 or higher recommended)
A basic understanding of CNN concepts and C# programming

Installation

Clone the repository:git clone https://github.com/yourusername/cnn-from-scratch.git


Navigate to the project directory:cd cnn-from-scratch


Build the project:dotnet build



Usage

Run the program:dotnet run


Follow the console prompts to:
Select a model architecture (Simple or VGG11).
Choose an optimizer (SGD or Adam).
Pick a hyperparameter preset (learning rate, batch size, dropout rate, epochs).


The program will:
Download and extract the CIFAR-10 dataset.
Train the selected model on the dataset.
Display training progress (loss and accuracy per batch/epoch).
Evaluate the model on the test set and report validation accuracy.
Save the trained model to cifar10_model.json.



Example Output
CIFAR-10 CNN Training
=====================
Choose model architecture:
(s) Simple 3x(C-B-R-C-B-R-M)=>D=>R=>D=>S
(v) VGG11
s
Choose optimizer:
(s) SGD (with momentum)
(a) Adam
a
Select hyperparameter preset:
(1) Adam - Phase 3 (LR=0.001, Batch=32, Dropout=0.3, Epochs=50, Augmentation)
1
Loading CIFAR-10 dataset...
Loaded 50000 training images
Data range: [0.000, 1.000]
Starting training...
Epoch 1/50 (lr=1.000E-03)
  Batch 10/1563 (320/50000 samples) | Loss: 2.3026, Accuracy: 10.00%
  ...
Epoch 1 complete. Final Loss: 1.8923, Accuracy: 32.45%
...
Validation Accuracy: 55.67%

Project Structure

Program.cs: Entry point, handles user input, model selection, and training orchestration.
Trainer.cs: Manages the training and evaluation pipeline, including batch processing, forward/backward passes, and optimization.
Core/: Contains tensor operations (Tensor3D, Tensor4D) for data manipulation.
Layers/: Implements CNN layers (Convolution, BatchNorm, ReLU, MaxPool, Dropout, Dense).
Data/: Handles CIFAR-10 dataset downloading and loading.
Models/: Defines the SequentialModel class and model creation logic.
Training/: Includes optimizer implementations (SGD, Adam) and loss functions (CrossEntropyLoss).
Models/Serialization/: Supports saving and loading models to/from JSON.

Limitations

Designed for educational purposes, not optimized for production-level performance.
Limited to CIFAR-10 dataset; extending to other datasets requires additional data loaders.
No GPU acceleration; computations are CPU-based.
Basic data augmentation (only for Adam optimizer in some presets).

Contributing
Contributions are welcome! If you'd like to improve the code, add features, or fix bugs, please:

Fork the repository.
Create a feature branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add YourFeature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Inspired by the CIFAR-10 dataset and VGG architecture.
Built for learning and experimentation with deep learning concepts in C#.
