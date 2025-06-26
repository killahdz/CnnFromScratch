using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using CnnFromScratch.Models;
using CnnFromScratch.Models.Serialization;
using NUnit.Framework;
using System;
using System.IO;

namespace CnnFromScratch.Tests.Models
{
    public class ModelSerializerTests
    {
        private string _testFilePath = null!;
        private SequentialModel _model = null!;

        [SetUp]
        public void Setup()
        {
            // Create a unique temporary file for each test
            _testFilePath = Path.Combine(
                Path.GetTempPath(),
                $"test_model_{Guid.NewGuid()}.json"
            );
            _model = new SequentialModel();
        }

        [TearDown]
        public void Cleanup()
        {
            try
            {
                if (File.Exists(_testFilePath))
                {
                    // Ensure file handle is released before deletion
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    File.Delete(_testFilePath);
                }
            }
            catch (IOException)
            {
                // Log or handle cleanup failure
                Console.WriteLine($"Warning: Could not delete test file: {_testFilePath}");
            }
        }

        [Test]
        public void Save_SimpleModel_CreatesFile()
        {
            // Arrange
            _model.AddLayer(new Conv2DLayer(3, 16, 3));
            _model.AddLayer(new ReLULayer());

            // Act
            ModelSerializer.Save(_model, _testFilePath);

            // Assert
            Assert.That(File.Exists(_testFilePath), Is.True);
        }

        [Test]
        public void LoadSave_ComplexModel_PreservesArchitecture()
        {
            // Arrange
            const int inputChannels = 3;
            const int inputHeight = 32;
            const int inputWidth = 32;
            
            // Calculate dimensions through the network
            var convOut = DimensionCalculator.CalculateOutputDimensions(
                inputChannels, inputHeight, inputWidth,
                kernelSize: 3, outputChannels: 16, stride: 1, padding: 1);
            
            var poolOut = (
                channels: convOut.channels,
                height: DimensionCalculator.CalculatePoolOutputSize(convOut.height, 2, 2),
                width: DimensionCalculator.CalculatePoolOutputSize(convOut.width, 2, 2)
            );
            
            var flattenedSize = DimensionCalculator.CalculateFlattenedSize(
                poolOut.channels, poolOut.height, poolOut.width);

            // Create model with correct dimensions
            _model.AddLayer(new Conv2DLayer(3, 16, 3, 1, 1));
            _model.AddLayer(new ReLULayer());
            _model.AddLayer(new MaxPoolLayer(2, 2));
            _model.AddLayer(new DenseLayer(flattenedSize, 10));
            _model.AddLayer(new SoftmaxLayer());

            // Create test input with correct dimensions
            var input = new Tensor3D(3, 32, 32);
            input.Randomize(-1, 1);  // Use consistent seed for reproducibility

            // Save original output before serialization
            var originalOutput = _model.Forward(input);

            // Act
            ModelSerializer.Save(_model, _testFilePath);
            var loadedModel = ModelSerializer.Load(_testFilePath);

            // Assert - Verify model structure
            Assert.Multiple(() =>
            {
                Assert.That(loadedModel.Layers, Is.Not.Null, "Loaded model layers should not be null");
                Assert.That(loadedModel.Layers.Count, Is.EqualTo(_model.Layers.Count), 
                    "Loaded model should have same number of layers");

                for (int i = 0; i < _model.Layers.Count; i++)
                {
                    Assert.That(loadedModel.Layers[i], Is.TypeOf(_model.Layers[i].GetType()),
                        $"Layer {i} type mismatch");
                }
            });

            // Forward pass validation
            var loadedOutput = loadedModel.Forward(input);

            // Compare outputs
            Assert.Multiple(() =>
            {
                Assert.That(loadedOutput.Channels, Is.EqualTo(originalOutput.Channels));
                Assert.That(loadedOutput.Height, Is.EqualTo(originalOutput.Height));
                Assert.That(loadedOutput.Width, Is.EqualTo(originalOutput.Width));

                for (int c = 0; c < originalOutput.Channels; c++)
                    for (int h = 0; h < originalOutput.Height; h++)
                        for (int w = 0; w < originalOutput.Width; w++)
                            Assert.That(loadedOutput[c, h, w], 
                                Is.EqualTo(originalOutput[c, h, w]).Within(1e-5f),
                                $"Mismatch at position [{c},{h},{w}]");
            });
        }

        [Test]
        public void Save_NullModel_ThrowsException()
        {
            Assert.Throws<ArgumentNullException>(() => 
                ModelSerializer.Save(null!, _testFilePath));
        }

        [Test]
        public void Load_NonexistentFile_ThrowsException()
        {
            Assert.Throws<FileNotFoundException>(() => 
                ModelSerializer.Load("nonexistent.json"));
        }

        [Test]
        public void Save_InvalidPath_ThrowsException()
        {
            Assert.Throws<ArgumentException>(() => 
                ModelSerializer.Save(_model, ""));
        }

        [Test]
        public void LoadSave_PreservesWeights()
        {
            // Arrange
            var conv = new Conv2DLayer(1, 1, 2);
            var weights = new float[1, 1, 2, 2];
            weights[0, 0, 0, 0] = 1.0f;
            weights[0, 0, 0, 1] = 2.0f;
            weights[0, 0, 1, 0] = 3.0f;
            weights[0, 0, 1, 1] = 4.0f;
            var biases = new float[] { 0.5f };
            conv.SetWeightsAndBiases(weights, biases);
            _model.AddLayer(conv);

            // Act
            ModelSerializer.Save(_model, _testFilePath);
            var loadedModel = ModelSerializer.Load(_testFilePath);

            // Assert
            var input = new Tensor3D(1, 2, 2);
            input.Fill(1.0f);

            var originalOutput = _model.Forward(input);
            var loadedOutput = loadedModel.Forward(input);

            Assert.That(loadedOutput[0, 0, 0], 
                Is.EqualTo(originalOutput[0, 0, 0]).Within(1e-5f));
        }
    }
}