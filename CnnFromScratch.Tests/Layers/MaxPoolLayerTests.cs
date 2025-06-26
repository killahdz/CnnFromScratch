using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using NUnit.Framework;
using System;

namespace CnnFromScratch.Tests.Layers
{
    public class MaxPoolLayerTests                      {
        private MaxPoolLayer _pool = null!;
        private Tensor3D _input = null!;

        [SetUp]
        public void Setup()
        {
            _pool = new MaxPoolLayer(poolSize: 2, stride: 2);
            _input = new Tensor3D(channels: 1, height: 4, width: 4);
        }

        [Test]
        public void Constructor_InvalidParameters_ThrowsException()
        {
            Assert.Multiple(() =>
            {
                Assert.Throws<ArgumentException>(() => new MaxPoolLayer(0, 1));
                Assert.Throws<ArgumentException>(() => new MaxPoolLayer(2, 0));
                Assert.Throws<ArgumentException>(() => new MaxPoolLayer(-1, 1));
            });
        }

        [Test]
        public void Forward_SimpleInput_CorrectMaxPooling()
        {
            // Arrange
            _input[0, 0, 0] = 1.0f; _input[0, 0, 1] = 2.0f;
            _input[0, 1, 0] = 3.0f; _input[0, 1, 1] = 4.0f;
            _input[0, 2, 0] = 5.0f; _input[0, 2, 1] = 6.0f;
            _input[0, 3, 0] = 7.0f; _input[0, 3, 1] = 8.0f;

            // Act
            var output = _pool.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output.Channels, Is.EqualTo(1));
                Assert.That(output.Height, Is.EqualTo(2));
                Assert.That(output.Width, Is.EqualTo(2));
                Assert.That(output[0, 0, 0], Is.EqualTo(4.0f)); // Max of first 2x2 block
                Assert.That(output[0, 1, 0], Is.EqualTo(8.0f)); // Max of second 2x2 block
            });
        }

        [Test]
        public void Forward_CorrectOutputDimensions()
        {
            // Arrange
            var input = new Tensor3D(2, 5, 5);

            // Act
            var output = _pool.Forward(input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output.Channels, Is.EqualTo(input.Channels));
                Assert.That(output.Height, Is.EqualTo(2)); // (5 - 2) / 2 + 1
                Assert.That(output.Width, Is.EqualTo(2));  // (5 - 2) / 2 + 1
            });
        }

        [Test]
        public void Backward_SimpleGradient_CorrectPropagation()
        {
            // Arrange
            _input[0, 0, 0] = 1.0f; _input[0, 0, 1] = 2.0f;
            _input[0, 1, 0] = 3.0f; _input[0, 1, 1] = 4.0f;

            var output = _pool.Forward(_input);
            var outputGradient = new Tensor3D(1, 2, 2);
            outputGradient[0, 0, 0] = 1.0f;

            // Act
            var inputGradient = _pool.Backward(outputGradient);

            // Assert
            Assert.Multiple(() =>
            {
                // Gradient should flow to the position that had the max value
                Assert.That(inputGradient[0, 0, 0], Is.EqualTo(0.0f));
                Assert.That(inputGradient[0, 0, 1], Is.EqualTo(0.0f));
                Assert.That(inputGradient[0, 1, 0], Is.EqualTo(0.0f));
                Assert.That(inputGradient[0, 1, 1], Is.EqualTo(1.0f)); // Position of max value
            });
        }

        [Test]
        public void Backward_WithoutForward_ThrowsException()
        {
            // Arrange
            var outputGradient = new Tensor3D(1, 2, 2);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => _pool.Backward(outputGradient));
        }

        [Test]
        public void Forward_MultiChannel_CorrectPooling()
        {
            // Arrange
            var input = new Tensor3D(2, 4, 4);
            // Channel 1
            input[0, 0, 0] = 1.0f; input[0, 0, 1] = 2.0f;
            input[0, 1, 0] = 3.0f; input[0, 1, 1] = 4.0f;
            // Channel 2
            input[1, 0, 0] = 5.0f; input[1, 0, 1] = 6.0f;
            input[1, 1, 0] = 7.0f; input[1, 1, 1] = 8.0f;

            // Act
            var output = _pool.Forward(input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output[0, 0, 0], Is.EqualTo(4.0f)); // Max from channel 1
                Assert.That(output[1, 0, 0], Is.EqualTo(8.0f)); // Max from channel 2
            });
        }
    }
}