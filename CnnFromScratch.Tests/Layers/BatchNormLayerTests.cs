using NUnit.Framework;
using CnnFromScratch.Layers;
using CnnFromScratch.Core;
using System;
using System.Linq;

namespace CnnFromScratch.Tests.Layers
{
  
    [TestFixture]
    public class BatchNormLayerTests
    {
        [Test]
        public void Forward_ShouldNormalizeEachChannelToMeanZeroVarianceOne()
        {
            // Arrange
            var input = new Tensor4D(batchSize: 2, channels: 3, height: 2, width: 2);
            var layer = new BatchNormLayer();
            layer.SetTrainingMode(true);

            // Fill input with sequential values per channel
            int val = 1;
            for (int b = 0; b < input.BatchSize; b++)
                for (int c = 0; c < input.Channels; c++)
                    for (int h = 0; h < input.Height; h++)
                        for (int w = 0; w < input.Width; w++)
                            input[b, c, h, w] = val++;

            // Act
            var output = layer.Forward(input);

            // Assert each channel mean ~0 and variance ~1
            for (int c = 0; c < input.Channels; c++)
            {
                float sum = 0f;
                float sumSq = 0f;
                int count = 0;

                for (int b = 0; b < input.BatchSize; b++)
                    for (int h = 0; h < input.Height; h++)
                        for (int w = 0; w < input.Width; w++)
                        {
                            float x = output[b, c, h, w];
                            sum += x;
                            sumSq += x * x;
                            count++;
                        }

                float mean = sum / count;
                float variance = sumSq / count - mean * mean;

                Assert.That(mean, Is.EqualTo(0).Within(1e-4), $"Mean for channel {c} was {mean}");
                Assert.That(variance, Is.EqualTo(1).Within(1e-3), $"Variance for channel {c} was {variance}");
            }
        }

        [Test]
        public void Inference_ShouldUseRunningStatistics()
        {
            // Arrange
            var input = Tensor4D.Ones(batch: 2, channels: 1, height: 2, width: 2);
            var layer = new BatchNormLayer();
            layer.SetTrainingMode(true);

            // Simulate training to initialize running stats
            layer.Forward(input);

            // Switch to inference
            layer.SetTrainingMode(false);

            // Act
            var output = layer.Forward(input);

            // Assert: All output values should be finite
            var data = output.Flatten();
            foreach (var x in data)
            {
                Assert.False(float.IsNaN(x) || float.IsInfinity(x), $"Invalid value: {x}");
            }
        }

        [Test]
        public void Backward_ShouldReturnTensorWithSameShapeAsInput()
        {
            // Arrange
            var input = Tensor4D.Random(batch: 2, channels: 2, height: 3, width: 3);
            var layer = new BatchNormLayer();
            layer.SetTrainingMode(true);

            var output = layer.Forward(input);
            var outputGrad = Tensor4D.OnesLike(output);

            // Act
            var inputGrad = layer.Backward(outputGrad);

            // Assert
            Assert.AreEqual(input.BatchSize, inputGrad.BatchSize);
            Assert.AreEqual(input.Channels, inputGrad.Channels);
            Assert.AreEqual(input.Height, inputGrad.Height);
            Assert.AreEqual(input.Width, inputGrad.Width);
        }
    }


}