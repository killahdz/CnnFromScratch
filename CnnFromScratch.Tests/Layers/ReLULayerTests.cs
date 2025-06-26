using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using NUnit.Framework;

namespace CnnFromScratch.Tests.Layers
{
    public class ReLULayerTests
    {
        private ReLULayer _relu = null!;
        private Tensor3D _input = null!;

        [SetUp]
        public void Setup()
        {
            _relu = new ReLULayer();
            
            // Create a 2x2x2 tensor for testing
            _input = new Tensor3D(channels: 2, height: 2, width: 2);
        }

        [Test]
        public void Forward_PositiveValues_RemainUnchanged()
        {
            // Arrange
            _input[0, 0, 0] = 1.5f;
            _input[0, 1, 1] = 2.0f;
            _input[1, 0, 1] = 3.0f;
            _input[1, 1, 0] = 4.0f;

            // Act
            var output = _relu.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output[0, 0, 0], Is.EqualTo(1.5f));
                Assert.That(output[0, 1, 1], Is.EqualTo(2.0f));
                Assert.That(output[1, 0, 1], Is.EqualTo(3.0f));
                Assert.That(output[1, 1, 0], Is.EqualTo(4.0f));
            });
        }

        [Test]
        public void Forward_NegativeValues_BecomeZero()
        {
            // Arrange
            _input[0, 0, 0] = -1.5f;
            _input[0, 1, 1] = -2.0f;
            _input[1, 0, 1] = -3.0f;
            _input[1, 1, 0] = -4.0f;

            // Act
            var output = _relu.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output[0, 0, 0], Is.EqualTo(0.0f));
                Assert.That(output[0, 1, 1], Is.EqualTo(0.0f));
                Assert.That(output[1, 0, 1], Is.EqualTo(0.0f));
                Assert.That(output[1, 1, 0], Is.EqualTo(0.0f));
            });
        }

        [Test]
        public void Forward_MixedValues_CorrectlyHandled()
        {
            // Arrange
            _input[0, 0, 0] = -1.5f;  // Should become 0
            _input[0, 1, 1] = 2.0f;   // Should remain 2
            _input[1, 0, 1] = 0.0f;   // Should remain 0
            _input[1, 1, 0] = -3.0f;  // Should become 0

            // Act
            var output = _relu.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output[0, 0, 0], Is.EqualTo(0.0f));
                Assert.That(output[0, 1, 1], Is.EqualTo(2.0f));
                Assert.That(output[1, 0, 1], Is.EqualTo(0.0f));
                Assert.That(output[1, 1, 0], Is.EqualTo(0.0f));
            });
        }

        [Test]
        public void Backward_PositiveInputs_GradientsPropagated()
        {
            // Arrange
            _input[0, 0, 0] = 1.5f;   // Gradient should pass through (>0)
            _input[0, 1, 1] = 2.0f;   // Gradient should pass through (>0)
            _input[1, 0, 1] = -1.0f;  // Gradient should be blocked (?0)
            _input[1, 1, 0] = 0.0f;   // Gradient should be blocked (?0)

            var outputGradient = new Tensor3D(2, 2, 2);
            outputGradient.Fill(1.0f);

            // Act
            _relu.Forward(_input);  // Need to call forward first to store input
            var inputGradient = _relu.Backward(outputGradient);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(inputGradient[0, 0, 0], Is.EqualTo(1.0f));  // Passes through
                Assert.That(inputGradient[0, 1, 1], Is.EqualTo(1.0f));  // Passes through
                Assert.That(inputGradient[1, 0, 1], Is.EqualTo(0.0f));  // Blocked
                Assert.That(inputGradient[1, 1, 0], Is.EqualTo(0.0f));  // Blocked
            });
        }

        [Test]
        public void Backward_WithoutForward_ThrowsException()
        {
            // Arrange
            var outputGradient = new Tensor3D(2, 2, 2);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => _relu.Backward(outputGradient));
        }

        [Test]
        public void Backward_GradientScaling_CorrectlyPropagated()
        {
            // Arrange
            _input[0, 0, 0] = 1.0f;   // Positive input
            _input[0, 1, 1] = -1.0f;  // Negative input

            var outputGradient = new Tensor3D(2, 2, 2);
            outputGradient[0, 0, 0] = 2.0f;  // Different gradient values
            outputGradient[0, 1, 1] = 3.0f;

            // Act
            _relu.Forward(_input);
            var inputGradient = _relu.Backward(outputGradient);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(inputGradient[0, 0, 0], Is.EqualTo(2.0f));  // Gradient scaled and passed
                Assert.That(inputGradient[0, 1, 1], Is.EqualTo(0.0f));  // Blocked due to negative input
            });
        }
    }
}