using CnnFromScratch.Core;
using CnnFromScratch.Data;
using CnnFromScratch.Layers;
using CnnFromScratch.Models;
using CnnFromScratch.Training;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;

namespace CnnFromScratch.Tests.Training
{
    public class TrainerTests
    {
        private SequentialModel _model = null!;
        private TestOptimizer _optimizer = null!;
        private TestLoss _loss = null!;
        private TestDataLoader _dataLoader = null!;
        private Trainer _trainer = null!;

        [SetUp]
        public void Setup()
        {
            _model = new SequentialModel();
            _optimizer = new TestOptimizer();
            _loss = new TestLoss();
            _dataLoader = new TestDataLoader();

            // Calculate dimensions through the network:
            // Input: 32x32x3
            // Conv2D: 32x32x16 (padding=1)
            // MaxPool: 16x16x16
            // Final flattened size = 16 * 16 * 16 = 4096

            _model.AddLayer(new Conv2DLayer(
                inputChannels: 3,
                outputChannels: 16,
                kernelSize: 3,
                stride: 1,
                padding: 1));
            _model.AddLayer(new ReLULayer());
            _model.AddLayer(new MaxPoolLayer(2, 2));
            _model.AddLayer(new DenseLayer(
                inputSize: 16 * 16 * 16,  // Corrected input size
                outputSize: 10));
            _model.AddLayer(new SoftmaxLayer());

            _trainer = new Trainer(_model, _optimizer, _loss, batchSize: 2);
        }

        [Test]
        public void Constructor_NullParameters_ThrowsException()
        {
            Assert.Multiple(() =>
            {
                Assert.Throws<ArgumentNullException>(() => new Trainer(null!, _optimizer, _loss));
                Assert.Throws<ArgumentNullException>(() => new Trainer(_model, null!, _loss));
                Assert.Throws<ArgumentNullException>(() => new Trainer(_model, _optimizer, null!));
            });
        }

        [Test]
        public void Train_ProcessesBatchesCorrectly()
        {
            // Arrange
            var trainData = CreateTestTrainingData();
            _dataLoader.SetTrainingData(trainData.Images, trainData.Labels);

            // Act
            _trainer.Train(_dataLoader, epochs: 1, learningRate: 0.01f);

            // Assert
            Assert.That(_optimizer.UpdateCount, Is.GreaterThan(0), "Optimizer should be called");
            Assert.That(_loss.CalculateCallCount, Is.GreaterThan(0), "Loss calculation should be called");
        }

        [Test]
        public void Evaluate_CalculatesAccuracyCorrectly()
        {
            // Arrange
            const int numClasses = 10;
            
            // Verify model configuration
            var denseLayer = _model.Layers.OfType<DenseLayer>().Last();
            Assert.That(denseLayer.OutputSize, Is.EqualTo(numClasses), 
                "Dense layer output size should match number of classes");

            // Create test data
            var testData = CreateTestTrainingData(numSamples: 4, numClasses: numClasses);
            _dataLoader.SetTestData(testData.Images, testData.Labels);

            // Create new trainer with fresh instances
            _loss = new TestLoss(numClasses);
            _optimizer = new TestOptimizer();
            _trainer = new Trainer(_model, _optimizer, _loss, batchSize: 2);

            // Act
            float accuracy = _trainer.Evaluate(_dataLoader);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(accuracy, Is.GreaterThanOrEqualTo(0.0f), "Accuracy should not be negative");
                Assert.That(accuracy, Is.LessThanOrEqualTo(1.0f), "Accuracy should not exceed 1.0");
                Assert.That(_loss.CalculateCallCount, Is.GreaterThan(0), "Loss should be calculated");
            });
        }

        private (Tensor3D[] Images, int[] Labels) CreateTestTrainingData(int numSamples = 4, int numClasses = 10)
        {
            // Create test images
            var images = new Tensor3D[numSamples];
            var labels = new int[numSamples];
            var random = new Random(42); // Fixed seed for reproducibility

            for (int i = 0; i < numSamples; i++)
            {
                images[i] = CreateTestImage();
                labels[i] = random.Next(numClasses); // Ensure labels are within valid range
            }

            return (images, labels);
        }

        private Tensor3D CreateTestImage()
        {
            var image = new Tensor3D(3, 32, 32); // CIFAR-10 format
            
            // Initialize with small random values for numerical stability
            var random = new Random();
            for (int c = 0; c < 3; c++)
                for (int h = 0; h < 32; h++)
                    for (int w = 0; w < 32; w++)
                        image[c, h, w] = (float)(random.NextDouble() * 0.1); // Small values

            return image;
        }

        [Test]
        public void Forward_DimensionsAreCorrect()
        {
            // Arrange
            var testImage = CreateTestImage();

            // Act & Assert - Check dimensions through the network
            var output = testImage;
            foreach (var layer in _model.Layers)
            {
                output = layer.Forward(output);
                Console.WriteLine($"Layer {layer.GetType().Name}: {output.Channels}x{output.Height}x{output.Width}");
            }

            // Final output should be 1x1x10 (10 classes in CIFAR-10)
            Assert.Multiple(() =>
            {
                Assert.That(output.Channels, Is.EqualTo(1), "Output should have 1 channel");
                Assert.That(output.Height, Is.EqualTo(1), "Output height should be 1");
                Assert.That(output.Width, Is.EqualTo(10), "Output width should be 10 (number of classes)");
            });
        }

        // Test implementations
        private class TestOptimizer : IOptimizer
        {
            public int UpdateCount { get; private set; }

            public void UpdateLayer(ILayer layer, float learningRate)
            {
                UpdateCount++;
            }
        }

        private class TestLoss : ILoss
        {
            public int CalculateCallCount { get; private set; }
            public int GradientCallCount { get; private set; }

            public TestLoss(int numClasses = 10)
            {
                // Optionally use numClasses for initialization if needed
            }

            public float Calculate(Tensor3D predicted, Tensor3D actual)
            {
                CalculateCallCount++;
                return 1.0f;
            }

            public Tensor3D Gradient(Tensor3D predicted, Tensor3D actual)
            {
                GradientCallCount++;
                return new Tensor3D(predicted.Channels, predicted.Height, predicted.Width);
            }
        }

        private class TestDataLoader : DataLoader
        {
            private (Tensor3D[] Images, int[] Labels) _trainingData;
            private (Tensor3D[] Images, int[] Labels) _testData;

            public TestDataLoader()
            {
                // Initialize with empty data
                _trainingData = (Array.Empty<Tensor3D>(), Array.Empty<int>());
                _testData = (Array.Empty<Tensor3D>(), Array.Empty<int>());
            }

            public void SetTrainingData(Tensor3D[] images, int[] labels)
            {
                _trainingData = (images, labels);
            }

            public void SetTestData(Tensor3D[] images, int[] labels)
            {
                _testData = (images, labels);
            }

            public override (Tensor3D[] Images, int[] Labels) LoadCifar10(bool isTraining = true)
            {
                return isTraining ? _trainingData : _testData;
            }

            public override IEnumerable<(Tensor3D[] Batch, int[] Labels)> GetBatches(int batchSize, bool shuffle = true)
            {
                var data = LoadCifar10(true);
                
                if (data.Images.Length == 0)
                    yield break;

                var indices = Enumerable.Range(0, data.Images.Length).ToArray();
                if (shuffle)
                {
                    var random = new Random();
                    indices = indices.OrderBy(x => random.Next()).ToArray();
                }

                for (int i = 0; i < data.Images.Length; i += batchSize)
                {
                    var batchIndices = indices.Skip(i).Take(batchSize).ToArray();
                    yield return (
                        batchIndices.Select(idx => data.Images[idx]).ToArray(),
                        batchIndices.Select(idx => data.Labels[idx]).ToArray()
                    );
                }
            }
        }
    }
}