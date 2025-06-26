using CnnFromScratch.Core;
using CnnFromScratch.Layers;

namespace CnnFromScratch.Tests.Layers
{
    public class DenseLayerTests
    {
        private DenseLayer _dense = null!;
        private Tensor3D _input = null!;

        [SetUp]
        public void Setup()
        {
            // Create a dense layer with 4 inputs and 2 outputs
            _dense = new DenseLayer(inputSize: 4, outputSize: 2);

            // Create a 1x2x2 input tensor (which will flatten to 4 values)
            _input = new Tensor3D(channels: 1, height: 2, width: 2);
        }

        [Test]
        public void Constructor_InvalidParameters_ThrowsException()
        {
            Assert.Multiple(() =>
            {
                Assert.Throws<ArgumentException>(() => new DenseLayer(0, 1));
                Assert.Throws<ArgumentException>(() => new DenseLayer(1, 0));
                Assert.Throws<ArgumentException>(() => new DenseLayer(-1, 1));
            });
        }

        [Test]
        public void Forward_CorrectOutputDimensions()
        {
            // Act
            var output = _dense.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output.Channels, Is.EqualTo(1));
                Assert.That(output.Height, Is.EqualTo(1));
                Assert.That(output.Width, Is.EqualTo(2)); // OutputSize
            });
        }

        [Test]
        public void Forward_SimpleInput_CorrectOutput()
        {
            // Arrange
            var simpleLayer = new DenseLayer(2, 1);
            var simpleInput = new Tensor3D(1, 1, 2);
            simpleInput[0, 0, 0] = 1.0f;
            simpleInput[0, 0, 1] = 2.0f;

            // Set weights and biases manually for predictable output
            var weights = new Matrix(1, 2); // outputSize x inputSize
            weights[0, 0] = 0.5f;
            weights[0, 1] = 0.5f;
            var biases = new float[1] { 0.0f };
            simpleLayer.SetWeightsAndBiases(weights, biases);

            // Act
            var output = simpleLayer.Forward(simpleInput);

            // Assert
            // Expected: 1.0 * 0.5 + 2.0 * 0.5 = 1.5
            Assert.That(output[0, 0, 0], Is.EqualTo(1.5f).Within(1e-5f));
        }

        [Test]
        public void Forward_InputSizeMismatch_ThrowsException()
        {
            // Arrange
            var wrongInput = new Tensor3D(2, 2, 2); // 8 values instead of 4

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _dense.Forward(wrongInput));
        }

        [Test]
        public void Backward_CorrectGradientDimensions()
        {
            // Arrange
            var output = _dense.Forward(_input);
            var outputGradient = new Tensor3D(1, 1, 2);
            outputGradient.Fill(1.0f);

            // Act
            var inputGradient = _dense.Backward(outputGradient);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(inputGradient.Channels, Is.EqualTo(1));
                Assert.That(inputGradient.Height, Is.EqualTo(1));
                Assert.That(inputGradient.Width, Is.EqualTo(4)); // InputSize
            });
        }

        [Test]
        public void Backward_SimpleGradient_CorrectPropagation()
        {
            // Arrange
            var simpleLayer = new DenseLayer(2, 1);
            var simpleInput = new Tensor3D(1, 1, 2);
            simpleInput[0, 0, 0] = 1.0f;
            simpleInput[0, 0, 1] = 2.0f;

            // Forward pass
            simpleLayer.Forward(simpleInput);

            // Create output gradient
            var outputGradient = new Tensor3D(1, 1, 1);
            outputGradient[0, 0, 0] = 1.0f;

            // Act
            var inputGradient = simpleLayer.Backward(outputGradient);

            // Assert
            Assert.That(simpleLayer.BiasGradients[0], Is.EqualTo(1.0f));
        }

        [Test]
        public void Backward_WithoutForward_ThrowsException()
        {
            // Arrange
            var outputGradient = new Tensor3D(1, 1, 2);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => _dense.Backward(outputGradient));
        }

        [Test]
        public void Forward_BiasAddition_CorrectOutput()
        {
            // Arrange
            var simpleLayer = new DenseLayer(1, 1);
            var simpleInput = new Tensor3D(1, 1, 1);
            simpleInput[0, 0, 0] = 1.0f;

            // Set weights and bias manually using the proper method
            var weights = new Matrix(1, 1); // outputSize x inputSize
            weights[0, 0] = 2.0f;
            var biases = new float[1] { 0.5f };
            simpleLayer.SetWeightsAndBiases(weights, biases);

            // Act
            var output = simpleLayer.Forward(simpleInput);

            // Assert
            // Expected: 1.0 * 2.0 + 0.5 = 2.5
            Assert.That(output[0, 0, 0], Is.EqualTo(2.5f).Within(1e-5f));
        }

        private float CalculateMean(Matrix matrix)
        {
            float sum = 0;
            int count = matrix.Rows * matrix.Cols;

            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Cols; j++)
                    sum += matrix[i, j];

            return sum / count;
        }

        private float CalculateStdDev(Matrix matrix, float mean)
        {
            float sumSquaredDiff = 0;
            int count = matrix.Rows * matrix.Cols;

            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Cols; j++)
                {
                    float diff = matrix[i, j] - mean;
                    sumSquaredDiff += diff * diff;
                }

            return (float)Math.Sqrt(sumSquaredDiff / count);
        }

        [Test]
        public void Constructor_WeightInitialization_HasCorrectStatistics()
        {
            // Arrange
            var layer = new DenseLayer(100, 50);
            var weights = layer.GetWeights(); // Need to add getter method

            // Assert
            Assert.Multiple(() =>
            {
                // He initialization: std = sqrt(2/input_size)
                float expectedStd = (float)Math.Sqrt(2.0 / 100);
                float actualMean = CalculateMean(weights);
                float actualStd = CalculateStdDev(weights, actualMean);

                Assert.That(actualMean, Is.EqualTo(0.0f).Within(0.1f), "Weights should be zero-centered");
                Assert.That(actualStd, Is.EqualTo(expectedStd).Within(0.1f), "Weights should follow He initialization");
            });
        }

        [Test]
        public void Backward_WeightGradients_CorrectComputation()
        {
            // Arrange
            var layer = new DenseLayer(2, 1);
            var input = new Tensor3D(1, 1, 2);
            input[0, 0, 0] = 1.0f;
            input[0, 0, 1] = 2.0f;

            var weights = new Matrix(1, 2);
            weights[0, 0] = 0.5f;
            weights[0, 1] = 0.5f;
            var biases = new float[1] { 0.0f };
            layer.SetWeightsAndBiases(weights, biases);

            // Forward pass
            var output = layer.Forward(input);
            
            // Backward pass
            var outputGradient = new Tensor3D(1, 1, 1);
            outputGradient[0, 0, 0] = 1.0f;
            layer.Backward(outputGradient);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(layer.WeightGradients[0, 0], Is.EqualTo(1.0f), "Gradient for first weight");
                Assert.That(layer.WeightGradients[0, 1], Is.EqualTo(2.0f), "Gradient for second weight");
            });
        }

        [Test]
        public void Backward_InputGradients_CorrectComputation()
        {
            // Arrange
            var layer = new DenseLayer(2, 2);
            var input = new Tensor3D(1, 1, 2);
            input[0, 0, 0] = 1.0f;
            input[0, 0, 1] = 2.0f;

            var weights = new Matrix(2, 2);
            weights[0, 0] = 0.5f; weights[0, 1] = 0.3f;
            weights[1, 0] = 0.2f; weights[1, 1] = 0.4f;
            var biases = new float[2] { 0.0f, 0.0f };
            layer.SetWeightsAndBiases(weights, biases);

            // Forward pass
            layer.Forward(input);

            // Backward pass with unit gradient
            var outputGradient = new Tensor3D(1, 1, 2);
            outputGradient[0, 0, 0] = 1.0f;
            outputGradient[0, 0, 1] = 1.0f;
            var inputGradient = layer.Backward(outputGradient);

            // Assert
            Assert.Multiple(() =>
            {
                // Input gradients are dot product of weights transpose and output gradient
                Assert.That(inputGradient[0, 0, 0], Is.EqualTo(0.7f).Within(1e-5f));  // 0.5 + 0.2
                Assert.That(inputGradient[0, 0, 1], Is.EqualTo(0.7f).Within(1e-5f));  // 0.3 + 0.4
            });
        }

        [Test]
        public void Forward_MultiChannelInput_CorrectFlattening()
        {
            // Arrange
            var layer = new DenseLayer(inputSize: 6, outputSize: 1);
            var input = new Tensor3D(channels: 2, height: 1, width: 3);
            
            // Fill input with sequential values
            int value = 1;
            for (int c = 0; c < 2; c++)
                for (int w = 0; w < 3; w++)
                    input[c, 0, w] = value++;

            // Set weights to 1 for easy verification
            var weights = new Matrix(1, 6);
            for (int i = 0; i < 6; i++)
                weights[0, i] = 1.0f;
            var biases = new float[1] { 0.0f };
            layer.SetWeightsAndBiases(weights, biases);

            // Act
            var output = layer.Forward(input);

            // Assert
            // Sum should be 1+2+3+4+5+6 = 21
            Assert.That(output[0, 0, 0], Is.EqualTo(21.0f).Within(1e-5f));
        }
    }
}