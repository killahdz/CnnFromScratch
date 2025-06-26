using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using NUnit.Framework;
using System;

namespace CnnFromScratch.Tests.Layers
{
    public class SoftmaxLayerTests
    {
        private SoftmaxLayer _softmax = null!;
        private Tensor3D _input = null!;

        [SetUp]
        public void Setup()
        {
            _softmax = new SoftmaxLayer();
            // Create a simple 1-channel 1x3 input (representing 3 class scores)
            _input = new Tensor3D(channels: 1, height: 1, width: 3);
        }

        [Test]
        public void Forward_SimpleInput_OutputSumsToOne()
        {
            // Arrange
            _input[0, 0, 0] = 1.0f;
            _input[0, 0, 1] = 2.0f;
            _input[0, 0, 2] = 3.0f;

            // Act
            var output = _softmax.Forward(_input);

            // Assert
            float sum = 0;
            for (int w = 0; w < output.Width; w++)
            {
                sum += output[0, 0, w];
            }
            Assert.That(sum, Is.EqualTo(1.0f).Within(1e-6f));
        }

        [Test]
        public void Forward_LargeValues_NoNumericalOverflow()
        {
            // Arrange
            _input[0, 0, 0] = 100.0f;
            _input[0, 0, 1] = 100.0f;
            _input[0, 0, 2] = 100.0f;

            // Act
            var output = _softmax.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output[0, 0, 0], Is.EqualTo(1.0f / 3.0f).Within(1e-6f));
                Assert.That(output[0, 0, 1], Is.EqualTo(1.0f / 3.0f).Within(1e-6f));
                Assert.That(output[0, 0, 2], Is.EqualTo(1.0f / 3.0f).Within(1e-6f));
            });
        }

        [Test]
        public void Forward_HighlyDifferentValues_CorrectProbabilities()
        {
            // Arrange
            _input[0, 0, 0] = 0.0f;
            _input[0, 0, 1] = 10.0f;
            _input[0, 0, 2] = -10.0f;

            // Act
            var output = _softmax.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output[0, 0, 1], Is.GreaterThan(0.99f));
                Assert.That(output[0, 0, 0], Is.LessThan(0.01f));
                Assert.That(output[0, 0, 2], Is.LessThan(0.01f));
            });
        }

        [Test]
        public void Forward_MultiplePixels_IndependentSoftmax()
        {
            // Arrange
            var input = new Tensor3D(1, 2, 3);
            for (int h = 0; h < 2; h++)
                for (int w = 0; w < 3; w++)
                {
                    input[0, h, w] = w + 1.0f; // Values 1.0, 2.0, 3.0
                }

            // Act
            var output = _softmax.Forward(input);

            // Assert
            for (int h = 0; h < 2; h++)
            {
                float sum = 0;
                for (int w = 0; w < 3; w++)
                {
                    sum += output[0, h, w];
                }
                Assert.That(sum, Is.EqualTo(1.0f).Within(1e-6f));
            }
        }

        [Test]
        public void Backward_SimpleGradient_CorrectPropagation()
        {
            // Arrange
            _input[0, 0, 0] = 1.0f;
            _input[0, 0, 1] = 2.0f;
            _input[0, 0, 2] = 3.0f;

            var output = _softmax.Forward(_input);

            var outputGradient = new Tensor3D(1, 1, 3);
            outputGradient[0, 0, 0] = 1.0f;
            outputGradient[0, 0, 1] = 0.0f;
            outputGradient[0, 0, 2] = 0.0f;

            // Act
            var inputGradient = _softmax.Backward(outputGradient);

            // Assert
            float sum = 0;
            for (int w = 0; w < inputGradient.Width; w++)
            {
                sum += inputGradient[0, 0, w];
            }
            Assert.That(sum, Is.EqualTo(0.0f).Within(1e-6f));
        }

        [Test]
        public void Backward_WithoutForward_ThrowsException()
        {
            // Arrange
            var outputGradient = new Tensor3D(1, 1, 3);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => _softmax.Backward(outputGradient));
        }
    }
}