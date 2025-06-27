# 🚀 CNN from Scratch 🚀
## ✨ Overview
This project presents a C# implementation of a Convolutional Neural Network (CNN) built entirely from the ground up! 🤯 My primary goal was educational: to help developers and learners dive deep into the inner workings of CNNs with a clear, modular, and extensible codebase.

It includes a complete training pipeline for the CIFAR-10 dataset, supporting both a simple architecture and a VGG11-inspired design, alongside options for SGD and Adam optimizers. You'll find core CNN components like convolutional layers, batch normalization, ReLU activation, max pooling, dropout, and dense layers, all crafted for readability and detailed documentation. It's truly a valuable resource for anyone studying deep learning fundamentals! 💡

## 🌟 Features
Custom CNN Implementation:

Core layers (Convolution, BatchNorm, ReLU, MaxPool, Dropout, Dense) built from scratch.

CIFAR-10 Dataset Support:

Automatically downloads and processes the CIFAR-10 dataset for training and evaluation. 🖼️

Flexible Model Architectures:

Simple CNN: 3x(Conv-BatchNorm-ReLU-Conv-BatchNorm-ReLU-MaxPool) → Dense → ReLU → Dense → Softmax

VGG11-inspired CNN: For more complex tasks.

Optimizer Options:

Stochastic Gradient Descent (SGD): With momentum and gradient clipping.

Adam Optimizer: For adaptive learning.

Hyperparameter Tuning:

Configurable learning rate, batch size, dropout rate, and epochs via console prompts. ⚙️

Comprehensive Training Pipeline:

Supports batch processing, learning rate decay, and validation accuracy tracking.

Model Serialization:

Save trained models to JSON for easy reuse. 💾

Robust Error Handling:

Thorough checks for data loading and model configuration. ✅

## 🎯 Purpose
This project was developed to foster a deeper understanding of CNNs by implementing them without relying on high-level frameworks like TensorFlow or PyTorch. It's designed as an educational tool for students, developers, and enthusiasts eager to learn how CNNs function "under the hood" – including forward/backward passes, intricate gradient computation, and various optimization techniques. 🧠

## 🚀 Getting Started
Prerequisites
.NET SDK (version 6.0 or higher recommended)

A basic understanding of CNN concepts and C# programming. 💻

## 🛠️ Installation
Clone the repository:
Build the project:

## 🏃 Usage
Run the program:

Follow the console prompts to:
- Select a model architecture (Simple or VGG11).
- Choose an optimizer (SGD or Adam).
- Pick a hyperparameter preset (learning rate, batch size, dropout rate, epochs).
The program will then:
- Download and extract the CIFAR-10 dataset. 📥
- Train the selected model on the dataset.
- Display training progress (loss and accuracy per batch/epoch).
- Evaluate the model on the test set and report validation accuracy.
- Save the trained model to cifar10_model.json. 📈

## 📂 Project Structure
- Program.cs: Entry point, handles user input, model selection, and training orchestration.
- Trainer.cs: Manages the training and evaluation pipeline, including batch processing, forward/backward passes, and optimization.
- Core/: Contains tensor operations (Tensor3D, Tensor4D) for data manipulation.
- Layers/: Implements CNN layers (Convolution, BatchNorm, ReLU, MaxPool, Dropout, Dense).
- Data/: Handles CIFAR-10 dataset downloading and loading.
- Models/: Defines the SequentialModel class and model creation logic.
- Training/: Includes optimizer implementations (SGD, Adam) and loss functions (CrossEntropyLoss).
- Models/Serialization/: Supports saving and loading models to/from JSON.

## ⚠️ Limitations
Educational Focus: Designed for learning, not optimized for production-level performance.

CIFAR-10 Specific: Limited to the CIFAR-10 dataset; extending to others requires additional data loaders.

CPU-Based: No GPU acceleration; all computations are CPU-based. 🐢

Basic Data Augmentation: Only available for the Adam optimizer in some presets.

## 👋 Contributions
Contributions are always welcome! If you'd like to improve the code, add features, or fix bugs, please feel free to:

Fork the repository.

Create a feature branch (git checkout -b feature/YourFeature).

Commit your changes (git commit -m 'Add YourFeature').

Push to the branch (git push origin feature/YourFeature).

Open a pull request. 🤝

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for full details.

## 🙏 Acknowledgments
Inspired by the Global AI Community.

Built purely for learning and experimentation with deep learning concepts in C#.

∞ Daniel Kereama ∞
