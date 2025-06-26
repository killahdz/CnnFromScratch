using CnnFromScratch.Core;
using CnnFromScratch.Data;
using CnnFromScratch.Layers;
using CnnFromScratch.Models;
using System;
using System.Linq;
using System.Collections.Generic; // Add this namespace for List<T>

namespace CnnFromScratch.Training
{
    /// <summary>
    /// Handles the training and evaluation of neural network models.
    /// Coordinates data batching, forward/backward passes, loss calculation,
    /// and optimization updates.
    /// </summary>
    public class Trainer
    {
        private readonly SequentialModel _model;
        private readonly IOptimizer _optimizer;
        private readonly ILoss _loss;
        private readonly int _batchSize;
        private readonly int _numClasses;

        // Training metrics
        public float CurrentLoss { get; private set; }
        public float CurrentAccuracy { get; private set; }

        public Trainer(SequentialModel model, IOptimizer optimizer, ILoss loss, int batchSize = 32)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _optimizer = optimizer ?? throw new ArgumentNullException(nameof(optimizer));
            _loss = loss ?? throw new ArgumentNullException(nameof(loss));
            _batchSize = batchSize;

            // Get number of classes from the model's final Dense layer
            var denseLayer = model.Layers.OfType<DenseLayer>().LastOrDefault() ??
                throw new ArgumentException("Model must contain a Dense layer for classification");
            _numClasses = denseLayer.OutputSize;

            // Validate number of classes
            if (_numClasses <= 0)
                throw new ArgumentException("Number of classes must be positive");
        }

        public void Train(DataLoader data, int epochs, float learningRate)
        {
            var (trainImages, trainLabels) = data.LoadCifar10(isTraining: true);
            int datasetSize = trainImages.Length;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float epochLoss = 0;
                int correctPredictions = 0;
                int totalSamples = 0;
                int batchIndex = 0;

                var batches = data.GetBatches(_batchSize, shuffle: true).ToList();
                int totalBatches = batches.Count;

                Console.WriteLine($"\nEpoch {epoch + 1}/{epochs} started...");

                foreach (var (batchImages, batchLabels) in batches)
                {
                    var (batchLoss, batchAccuracy) = TrainBatch(batchImages, batchLabels, learningRate);

                    epochLoss += batchLoss * batchImages.Length;
                    correctPredictions += (int)(batchAccuracy * batchImages.Length);
                    totalSamples += batchImages.Length;

                    // Update metrics
                    CurrentLoss = epochLoss / totalSamples;
                    CurrentAccuracy = (float)correctPredictions / totalSamples;

                    batchIndex++;

                    if (batchIndex % 10 == 0 || batchIndex == totalBatches)
                    {
                        Console.WriteLine($"  Batch {batchIndex}/{totalBatches} ({totalSamples}/{datasetSize} samples) | " +
                                          $"Loss: {CurrentLoss:F4}, Accuracy: {CurrentAccuracy:P2}");
                    }
                }

                Console.WriteLine($"Epoch {epoch + 1} complete. Final Loss: {CurrentLoss:F4}, Accuracy: {CurrentAccuracy:P2}");
            }
        }
     
        private (float Loss, float Accuracy) TrainBatch(Tensor3D[] batchImages, int[] batchLabels, float learningRate)
        {
            int batchSize = batchImages.Length;

            // Convert Tensor3D[] to Tensor4D batch
            var inputBatch = new Tensor4D(batchSize, batchImages[0].Channels, batchImages[0].Height, batchImages[0].Width);
            for (int i = 0; i < batchSize; i++)
                inputBatch.SetSlice(i, batchImages[i]);

            // Forward pass on batch
            var outputsBatch = _model.Forward(inputBatch);

            // Prepare one-hot encoded batch labels as Tensor4D
            var targetBatch = new Tensor4D(batchSize, 1, 1, _numClasses);
            for (int i = 0; i < batchSize; i++)
            {
                var oneHot = new Tensor3D(1, 1, _numClasses);
                oneHot.Fill(0);
                oneHot[0, 0, batchLabels[i]] = 1;
                targetBatch.SetSlice(i, oneHot);
            }

            // Calculate batch loss
            float batchLoss = 0;
            for (int i = 0; i < batchSize; i++)
            {
                var output = outputsBatch.GetSlice(i);
                var target = targetBatch.GetSlice(i);
                batchLoss += _loss.Calculate(output, target);
            }
            batchLoss /= batchSize;

            // Calculate batch accuracy
            int correctPredictions = 0;
            for (int i = 0; i < batchSize; i++)
            {
                var output = outputsBatch.GetSlice(i);
                int predictedClass = GetPredictedClass(output);
                if (predictedClass == batchLabels[i])
                    correctPredictions++;
            }
            float batchAccuracy = (float)correctPredictions / batchSize;

            // Compute gradient batch from loss function
            var gradientsBatch = new Tensor4D(batchSize, 1, 1, _numClasses);
            for (int i = 0; i < batchSize; i++)
            {
                var output = outputsBatch.GetSlice(i);
                var target = targetBatch.GetSlice(i);
                var grad = _loss.Gradient(output, target);
                gradientsBatch.SetSlice(i, grad);
            }

            // Clip gradients to prevent explosion
            ClipGradients(gradientsBatch, 1.0f);

            // Backward pass on batch gradients
            var inputGradients = _model.Backward(gradientsBatch);

            // Update weights of each layer using optimizer
            foreach (var layer in _model.Layers)
            {
                _optimizer.UpdateLayer(layer, learningRate);
            }

            return (batchLoss, batchAccuracy);
        }

        private void ClipGradients(Tensor4D gradients, float threshold)
        {
            for (int n = 0; n < gradients.BatchSize; n++)
                for (int c = 0; c < gradients.Channels; c++)
                    for (int h = 0; h < gradients.Height; h++)
                        for (int w = 0; w < gradients.Width; w++)
                        {
                            float value = gradients[n, c, h, w];
                            if (float.IsNaN(value))
                            {
                                gradients[n, c, h, w] = 0;
                                continue;
                            }
                            gradients[n, c, h, w] = Math.Max(Math.Min(value, threshold), -threshold);
                        }
        }

        //private void BackwardPass(Tensor3D gradients, float learningRate)
        //{
        //    var currentGradient = gradients;
        //    var layers = GetModelLayers();

        //    const float clipThreshold = 1.0f;

        //    // Backward pass through all layers
        //    for (int i = layers.Count - 1; i >= 0; i--)
        //    {
        //        // Clip gradients to prevent explosion
        //        ClipGradients(currentGradient, clipThreshold);

        //        currentGradient = layers[i].Backward(currentGradient);
        //        _optimizer.UpdateLayer(layers[i], learningRate);
        //    }
        //}

        //private void ClipGradients(Tensor3D gradients, float threshold)
        //{
        //    for (int c = 0; c < gradients.Channels; c++)
        //        for (int h = 0; h < gradients.Height; h++)
        //            for (int w = 0; w < gradients.Width; w++)
        //            {
        //                float value = gradients[c, h, w];
        //                if (float.IsNaN(value))
        //                {
        //                    gradients[c, h, w] = 0;
        //                    continue;
        //                }
        //                gradients[c, h, w] = Math.Max(Math.Min(value, threshold), -threshold);
        //            }
        //}

        private int GetPredictedClass(Tensor3D output)
        {
            int predictedClass = 0;
            float maxProbability = float.MinValue;

            // Assuming output is 1x1xN where N is number of classes
            for (int c = 0; c < _numClasses; c++)
            {
                float probability = output[0, 0, c];
                if (probability > maxProbability)
                {
                    maxProbability = probability;
                    predictedClass = c;
                }
            }

            return predictedClass;
        }

        private Tensor3D CreateOneHotEncoding(int label, int numClasses)
        {
            // Use the class's _numClasses field instead of the parameter
            if (label < 0 || label >= _numClasses)
                throw new ArgumentException($"Label {label} is outside valid range [0-{_numClasses-1}]");

            var oneHot = new Tensor3D(1, 1, _numClasses);
            oneHot.Fill(0);
            oneHot[0, 0, label] = 1;
            return oneHot;
        }

        //private List<ILayer> GetModelLayers()
        //{
        //    return _model.Layers.ToList();
        //}

        public (float Loss, float Accuracy) Evaluate(Tensor3D[] images, int[] labels)
        {
            if (images == null || labels == null)
                throw new ArgumentNullException(images == null ? nameof(images) : nameof(labels));
            
            if (images.Length != labels.Length)
                throw new ArgumentException("Number of images must match number of labels");

            float totalLoss = 0;
            int correctPredictions = 0;

            for (int i = 0; i < images.Length; i++)
            {
                var output = _model.Forward(images[i]);
                var target = CreateOneHotEncoding(labels[i], _numClasses);
                
                totalLoss += _loss.Calculate(output, target);
                
                int predictedClass = GetPredictedClass(output);
                if (predictedClass == labels[i])
                    correctPredictions++;
            }

            return (totalLoss / images.Length, (float)correctPredictions / images.Length);
        }

        public float Evaluate(DataLoader dataLoader)
        {
            var (testImages, testLabels) = dataLoader.LoadCifar10(isTraining: false);
            var (loss, accuracy) = Evaluate(testImages, testLabels);
            return accuracy;
        }
    }
}