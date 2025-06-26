using CnnFromScratch.Core;
using System;

namespace CnnFromScratch.Layers
{
    /// <summary>
    /// Softmax activation layer that converts raw scores to probabilities.
    /// Applies softmax function: exp(x_i) / sum(exp(x_j)) for all j
    /// 
    /// Key characteristics:
    /// - Converts scores to probabilities (values sum to 1)
    /// - Numerically stable implementation
    /// - Typically used as final layer for classification
    /// - No trainable parameters (weights or biases)
    /// </summary>
    public class SoftmaxLayer : ILayer
    {
        private Tensor3D? _lastInput;
        private Tensor3D? _lastOutput;

        public Tensor3D Forward(Tensor3D input)
        {
            _lastInput = input.Clone();
            var output = new Tensor3D(input.Channels, input.Height, input.Width);

            // For each channel and height (sample), apply softmax across width (classes)
            for (int c = 0; c < input.Channels; c++)
            {
                for (int h = 0; h < input.Height; h++)
                {
                    // Find max value for numerical stability
                    float maxVal = float.MinValue;
                    for (int w = 0; w < input.Width; w++)
                    {
                        maxVal = Math.Max(maxVal, input[c, h, w]);
                    }

                    // Compute exp(x - max) and sum
                    float sum = 0;
                    for (int w = 0; w < input.Width; w++)
                    {
                        float exp = (float)Math.Exp(input[c, h, w] - maxVal);
                        output[c, h, w] = exp;
                        sum += exp;
                    }

                    // Normalize by sum
                    for (int w = 0; w < input.Width; w++)
                    {
                        output[c, h, w] /= sum;
                    }
                }
            }

            _lastOutput = output.Clone();
            return output;
        }

        public Tensor3D Backward(Tensor3D outputGradient)
        {
            if (_lastOutput == null || _lastInput == null)
                throw new InvalidOperationException("Backward called before Forward");

            var inputGradient = new Tensor3D(_lastInput.Channels, _lastInput.Height, _lastInput.Width);

            // Compute gradient for each channel and height (sample)
            for (int c = 0; c < _lastInput.Channels; c++)
            {
                for (int h = 0; h < _lastInput.Height; h++)
                {
                    for (int i = 0; i < _lastInput.Width; i++)
                    {
                        float sum = 0;
                        for (int j = 0; j < _lastInput.Width; j++)
                        {
                            // Kronecker delta (?ij) - 1 if i==j, 0 otherwise
                            float delta = i == j ? 1 : 0;

                            // Gradient of softmax: si * (?ij - sj)
                            float softmaxGrad = _lastOutput[c, h, i] * (delta - _lastOutput[c, h, j]);
                            sum += outputGradient[c, h, j] * softmaxGrad;
                        }
                        inputGradient[c, h, i] = sum;
                    }
                }
            }

            return inputGradient;
        }

        /// <summary>
        /// Gets the weights of the layer. Since Softmax has no weights,
        /// returns an empty array.
        /// </summary>
        /// <returns>Empty array as Softmax has no weights</returns>
        public object GetWeights()
        {
            return Array.Empty<float>();
        }

        /// <summary>
        /// Gets the biases of the layer. Since Softmax has no biases,
        /// returns an empty array.
        /// </summary>
        /// <returns>Empty array as Softmax has no biases</returns>
        public float[] GetBiases()
        {
            return Array.Empty<float>();
        }

        /// <summary>
        /// Sets weights and biases for the layer. Since Softmax has no
        /// trainable parameters, this method is a no-op.
        /// </summary>
        /// <param name="weights">Ignored</param>
        /// <param name="biases">Ignored</param>
        public void SetWeightsAndBiases(object weights, float[] biases)
        {
            // No-op as Softmax has no weights or biases
        }
    }
}