using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using NUnit.Framework;
using System;

namespace CnnFromScratch.Tests.Layers
{
    public class Conv2DLayerTests
    {
        private Conv2DLayer _conv = null!;
        private Tensor3D _input = null!;

        [SetUp]
        public void Setup()
        {
            // Create a simple 3x3 convolution layer with 2 input channels and 3 output channels
            _conv = new Conv2DLayer(
                inputChannels: 2,
                outputChannels: 3,
                kernelSize: 3,
                stride: 1,
                padding: 1
            );

            // Create a 2-channel 4x4 input tensor
            _input = new Tensor3D(channels: 2, height: 4, width: 4);
        }

        [Test]
        public void Constructor_ValidParameters_CreatesLayer()
        {
            // Arrange & Act
            var layer = new Conv2DLayer(
                inputChannels: 3,
                outputChannels: 16,
                kernelSize: 3,
                stride: 2,
                padding: 1
            );

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(layer.InputChannels, Is.EqualTo(3));
                Assert.That(layer.OutputChannels, Is.EqualTo(16));
                Assert.That(layer.KernelSize, Is.EqualTo(3));
                Assert.That(layer.Stride, Is.EqualTo(2));
                Assert.That(layer.Padding, Is.EqualTo(1));
            });
        }

        [Test]
        public void Forward_InvalidInputChannels_ThrowsException()
        {
            // Arrange
            var invalidInput = new Tensor3D(channels: 3, height: 4, width: 4);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => _conv.Forward(invalidInput));
        }

        [Test]
        public void Forward_WithPadding_CorrectOutputDimensions()
        {
            // Arrange - using _input from Setup

            // Act
            var output = _conv.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output.Channels, Is.EqualTo(_conv.OutputChannels));
                Assert.That(output.Height, Is.EqualTo(_input.Height));    // Same size due to padding
                Assert.That(output.Width, Is.EqualTo(_input.Width));      // Same size due to padding
            });
        }

        [Test]
        public void Forward_WithoutPadding_CorrectOutputDimensions()
        {
            // Arrange
            var noPaddingConv = new Conv2DLayer(
                inputChannels: 2,
                outputChannels: 3,
                kernelSize: 3,
                padding: 0
            );

            // Act
            var output = noPaddingConv.Forward(_input);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output.Channels, Is.EqualTo(noPaddingConv.OutputChannels));
                Assert.That(output.Height, Is.EqualTo(_input.Height - 2)); // Reduced by kernel size - 1
                Assert.That(output.Width, Is.EqualTo(_input.Width - 2));   // Reduced by kernel size - 1
            });
        }

        [Test]
        public void Forward_WithStride_CorrectOutputDimensions()
        {
            // Arrange
            var strideConv = new Conv2DLayer(
                inputChannels: 2,
                outputChannels: 3,
                kernelSize: 3,
                stride: 2,
                padding: 1
            );

            // Act
            var output = strideConv.Forward(_input);

            // Assert
            int expectedHeight = (_input.Height + 2 * strideConv.Padding - strideConv.KernelSize) / strideConv.Stride + 1;
            int expectedWidth = (_input.Width + 2 * strideConv.Padding - strideConv.KernelSize) / strideConv.Stride + 1;

            Assert.Multiple(() =>
            {
                Assert.That(output.Channels, Is.EqualTo(strideConv.OutputChannels));
                Assert.That(output.Height, Is.EqualTo(expectedHeight));
                Assert.That(output.Width, Is.EqualTo(expectedWidth));
            });
        }

        [Test]
        public void Forward_SimpleInput_CorrectConvolution()
        {
            // Arrange
            var simpleConv = new Conv2DLayer(
                inputChannels: 1,
                outputChannels: 1,
                kernelSize: 2,
                stride: 1,
                padding: 0
            );

            var simpleInput = new Tensor3D(1, 2, 2);
            simpleInput[0, 0, 0] = 1;
            simpleInput[0, 0, 1] = 2;
            simpleInput[0, 1, 0] = 3;
            simpleInput[0, 1, 1] = 4;

            // Set weights and biases for predictable output
            var weights = new float[1, 1, 2, 2]; // [outputChannels, inputChannels, kernelHeight, kernelWidth]
            for (int kh = 0; kh < 2; kh++)
                for (int kw = 0; kw < 2; kw++)
                    weights[0, 0, kh, kw] = 1.0f;

            var biases = new float[1] { 0.0f };
            simpleConv.SetWeightsAndBiases(weights, biases);

            // Act
            var output = simpleConv.Forward(simpleInput);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(output.Channels, Is.EqualTo(1));
                Assert.That(output.Height, Is.EqualTo(1));
                Assert.That(output.Width, Is.EqualTo(1));
                // Convolution calculation:
                // 1*1 + 2*1 + 3*1 + 4*1 + 0(bias) = 10
                Assert.That(output[0, 0, 0], Is.EqualTo(10.0f).Within(1e-5f));
            });
        }

        [Test]
        public void Forward_SimpleInput_VerifyConvolutionComputation()
        {
            // Arrange
            var simpleConv = new Conv2DLayer(
                inputChannels: 1,
                outputChannels: 1,
                kernelSize: 2,
                stride: 1,
                padding: 0
            );

            var simpleInput = new Tensor3D(1, 2, 2);
            simpleInput[0, 0, 0] = 1;
            simpleInput[0, 0, 1] = 2;
            simpleInput[0, 1, 0] = 3;
            simpleInput[0, 1, 1] = 4;

            // Set specific weights for verification
            var weights = new float[1, 1, 2, 2];
            weights[0, 0, 0, 0] = 1.0f; // top-left
            weights[0, 0, 0, 1] = 2.0f; // top-right
            weights[0, 0, 1, 0] = 3.0f; // bottom-left
            weights[0, 0, 1, 1] = 4.0f; // bottom-right

            var biases = new float[1] { 1.0f };
            simpleConv.SetWeightsAndBiases(weights, biases);

            // Act
            var output = simpleConv.Forward(simpleInput);

            // Assert
            // Expected: (1*1 + 2*2 + 3*3 + 4*4) + 1(bias) = 30 + 1 = 31
            Assert.That(output[0, 0, 0], Is.EqualTo(31.0f).Within(1e-5f));
        }

        [Test]
        public void Backward_CorrectGradientDimensions()
        {
            // Arrange
            var output = _conv.Forward(_input);
            var outputGradient = new Tensor3D(
                channels: _conv.OutputChannels,
                height: output.Height,
                width: output.Width
            );
            outputGradient.Fill(1.0f);

            // Act
            var inputGradient = _conv.Backward(outputGradient);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(inputGradient.Channels, Is.EqualTo(_input.Channels));
                Assert.That(inputGradient.Height, Is.EqualTo(_input.Height));
                Assert.That(inputGradient.Width, Is.EqualTo(_input.Width));
            });
        }

        [Test]
        public void Backward_WithoutForward_ThrowsException()
        {
            // Arrange
            var outputGradient = new Tensor3D(3, 4, 4);

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => _conv.Backward(outputGradient));
        }
    }
}