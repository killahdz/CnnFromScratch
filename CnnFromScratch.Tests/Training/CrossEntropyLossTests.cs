using CnnFromScratch.Core;
using CnnFromScratch.Training;
using NUnit.Framework;
using System;

namespace CnnFromScratch.Tests.Training
{
    public class CrossEntropyLossTests
    {
        private CrossEntropyLoss _loss = null!;

        [SetUp]
        public void Setup()
        {
            _loss = new CrossEntropyLoss();
        }

        [Test]
        public void Calculate_PerfectPrediction_ReturnsZero()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 3);

            predicted[0, 0, 0] = 1.0f;
            predicted[0, 0, 1] = 0.0f;
            predicted[0, 0, 2] = 0.0f;

            actual[0, 0, 0] = 1.0f;
            actual[0, 0, 1] = 0.0f;
            actual[0, 0, 2] = 0.0f;

            // Act
            float loss = _loss.Calculate(predicted, actual);

            // Assert
            Assert.That(loss, Is.LessThan(1e-4f)); // or EqualTo(0f).Within(1e-5f)
        }

        [Test]
        public void Calculate_CompletelyWrong_ReturnsLargeValue()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 3);

            predicted[0, 0, 0] = 0.0f;
            predicted[0, 0, 1] = 1.0f;
            predicted[0, 0, 2] = 0.0f;

            actual[0, 0, 0] = 1.0f;
            actual[0, 0, 1] = 0.0f;
            actual[0, 0, 2] = 0.0f;

            // Act
            float loss = _loss.Calculate(predicted, actual);

            // Assert
            Assert.That(loss, Is.GreaterThan(5.0f));
        }

        [Test]
        public void Calculate_MultipleSamples_ReturnsAverageLoss()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 3);

            // Set predicted to perfect prediction (matching actual)
            predicted[0, 0, 0] = 1.0f;
            predicted[0, 0, 1] = 0.0f;
            predicted[0, 0, 2] = 0.0f;

            // Set actual (one-hot encoding)
            actual[0, 0, 0] = 1.0f;
            actual[0, 0, 1] = 0.0f;
            actual[0, 0, 2] = 0.0f;

            // Act
            float loss = _loss.Calculate(predicted, actual);

            // Assert
            // Cross-entropy loss for perfect prediction should be very close to 0
            Assert.That(loss, Is.EqualTo(0f).Within(1e-4f));
        }

        [Test]
        public void Calculate_WrongPrediction_ReturnsHighLoss()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 3);

            // Predicted distribution (after softmax)
            predicted[0, 0, 0] = 0.1f;
            predicted[0, 0, 1] = 0.8f; // High confidence in wrong class
            predicted[0, 0, 2] = 0.1f;

            // Actual (one-hot)
            actual[0, 0, 0] = 1.0f; // True class
            actual[0, 0, 1] = 0.0f;
            actual[0, 0, 2] = 0.0f;

            // Act
            float loss = _loss.Calculate(predicted, actual);

            // Assert
            // For predicted probability of 0.1 for true class:
            // -log(0.1) ≈ 2.30
            Assert.That(loss, Is.GreaterThan(2.0f), "Loss should be high for incorrect prediction");
        }

        [Test]
        public void Calculate_ValidatesInputDimensions()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 2); // Different number of classes

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _loss.Calculate(predicted, actual));
        }

        [Test]
        public void Gradient_PerfectPrediction_SmallGradients()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 3);

            predicted[0, 0, 0] = 1.0f;
            predicted[0, 0, 1] = 0.0f;
            predicted[0, 0, 2] = 0.0f;

            actual[0, 0, 0] = 1.0f;
            actual[0, 0, 1] = 0.0f;
            actual[0, 0, 2] = 0.0f;

            // Act
            var gradient = _loss.Gradient(predicted, actual);

            // Assert
            Assert.That(Math.Abs(gradient[0, 0, 0]), Is.LessThan(1.1f));
        }

        [Test]
        public void Gradient_CompletelyWrong_LargeGradients()
        {
            // Arrange
            var loss = new CrossEntropyLoss(withSoftmax: false); // Pure cross-entropy
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 3);

            predicted[0, 0, 0] = 0.0f;
            predicted[0, 0, 1] = 1.0f;
            predicted[0, 0, 2] = 0.0f;

            actual[0, 0, 0] = 1.0f;
            actual[0, 0, 1] = 0.0f;
            actual[0, 0, 2] = 0.0f;

            // Act
            var gradient = loss.Gradient(predicted, actual);

            // Assert
            Assert.That(Math.Abs(gradient[0, 0, 0]), Is.EqualTo(1e5f).Within(1f));
        }

        [Test]
        public void Gradient_MatchesPredictedDimensions()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 3);
            predicted.Fill(1.0f / 3.0f); // Uniform distribution
            actual[0, 0, 0] = 1.0f; // One-hot encoding

            // Act
            var gradient = _loss.Gradient(predicted, actual);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(gradient.Channels, Is.EqualTo(predicted.Channels));
                Assert.That(gradient.Height, Is.EqualTo(predicted.Height));
                Assert.That(gradient.Width, Is.EqualTo(predicted.Width));
            });
        }

        [Test]
        public void Gradient_CorrectValues()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 2);
            var actual = new Tensor3D(1, 1, 2);

            // Softmax output (probabilities sum to 1)
            predicted[0, 0, 0] = 0.6f;
            predicted[0, 0, 1] = 0.4f;

            // True label (one-hot)
            actual[0, 0, 0] = 1.0f;
            actual[0, 0, 1] = 0.0f;

            // Act
            var gradient = _loss.Gradient(predicted, actual);

            // Assert
            // Gradient should be (predicted - actual) / number_of_samples
            Assert.Multiple(() =>
            {
                Assert.That(gradient[0, 0, 0], Is.EqualTo(-0.4f).Within(1e-5f));  // (0.6 - 1.0)
                Assert.That(gradient[0, 0, 1], Is.EqualTo(0.4f).Within(1e-5f));   // (0.4 - 0.0)
            });
        }

        [Test]
        public void Calculate_MismatchedDimensions_ThrowsException()
        {
            // Arrange
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 2); // Different dimensions

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _loss.Calculate(predicted, actual));
            Assert.Throws<ArgumentException>(() => _loss.Gradient(predicted, actual));
        }

        [Test]
        public void Gradient_CompletelyWrong_CorrectGradient()
        {
            // Arrange
            var loss = new CrossEntropyLoss(withSoftmax: true); // With softmax
            var predicted = new Tensor3D(1, 1, 3);
            var actual = new Tensor3D(1, 1, 3);

            // When prediction is completely wrong:
            // One class has probability 1.0 when it should be 0.0
            predicted[0, 0, 0] = 0.0f;  // Should be 1.0
            predicted[0, 0, 1] = 1.0f;  // Should be 0.0
            predicted[0, 0, 2] = 0.0f;  // Should be 0.0

            actual[0, 0, 0] = 1.0f;
            actual[0, 0, 1] = 0.0f;
            actual[0, 0, 2] = 0.0f;

            // Act
            var gradient = loss.Gradient(predicted, actual);

            // Assert
            Assert.Multiple(() =>
            {
                // For the correct class (index 0):
                // gradient = 0.0 - 1.0 = -1.0 (maximum negative gradient)
                Assert.That(gradient[0, 0, 0], Is.EqualTo(-1.0f).Within(1e-5f), 
                    "Gradient for true class should be -1.0");

                // For the wrongly predicted class (index 1):
                // gradient = 1.0 - 0.0 = 1.0 (maximum positive gradient)
                Assert.That(gradient[0, 0, 1], Is.EqualTo(1.0f).Within(1e-5f), 
                    "Gradient for incorrect prediction should be 1.0");

                // For the other class (index 2):
                // gradient = 0.0 - 0.0 = 0.0 (no gradient)
                Assert.That(gradient[0, 0, 2], Is.EqualTo(0.0f).Within(1e-5f), 
                    "Gradient for unrelated class should be 0.0");
            });
        }
    }
}