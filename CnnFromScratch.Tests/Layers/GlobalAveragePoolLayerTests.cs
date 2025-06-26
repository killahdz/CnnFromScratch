using System;
using NUnit.Framework;
using CnnFromScratch.Core;
using CnnFromScratch.Layers;

namespace CnnFromScratch.Tests.Layers
{
    [TestFixture]
    internal class GlobalAveragePoolLayerTests
    {
        [Test]
        public void Forward_ComputesGlobalAverageCorrectly()
        {
            // Arrange
            var input = new Tensor3D(2, 2, 2);
            input[0, 0, 0] = 1;
            input[0, 0, 1] = 2;
            input[0, 1, 0] = 3;
            input[0, 1, 1] = 4; // avg = (1+2+3+4)/4 = 2.5

            input[1, 0, 0] = 10;
            input[1, 0, 1] = 20;
            input[1, 1, 0] = 30;
            input[1, 1, 1] = 40; // avg = (10+20+30+40)/4 = 25.0

            var layer = new GlobalAveragePoolLayer();

            // Act
            var output = layer.ForwardWithShapeTracking(input);

            // Assert
            Assert.That(output.Channels, Is.EqualTo(2));
            Assert.That(output.Height, Is.EqualTo(1));
            Assert.That(output.Width, Is.EqualTo(1));
            Assert.That(output[0, 0, 0], Is.EqualTo(2.5f).Within(0.001f));
            Assert.That(output[1, 0, 0], Is.EqualTo(25.0f).Within(0.001f));
        }

        [Test]
        public void Backward_DistributesGradientEqually()
        {
            // Arrange
            var input = new Tensor3D(1, 2, 2);
            input[0, 0, 0] = 1;
            input[0, 0, 1] = 1;
            input[0, 1, 0] = 1;
            input[0, 1, 1] = 1;

            var layer = new GlobalAveragePoolLayer();
            _ = layer.ForwardWithShapeTracking(input);

            var outputGrad = new Tensor3D(1, 1, 1);
            outputGrad[0, 0, 0] = 8.0f;

            // Act
            var grad = layer.Backward(outputGrad);

            // Assert
            float expected = 8.0f / 4.0f;
            Assert.That(grad[0, 0, 0], Is.EqualTo(expected).Within(0.001f));
            Assert.That(grad[0, 0, 1], Is.EqualTo(expected).Within(0.001f));
            Assert.That(grad[0, 1, 0], Is.EqualTo(expected).Within(0.001f));
            Assert.That(grad[0, 1, 1], Is.EqualTo(expected).Within(0.001f));
        }
    }
}
