using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using System;
using System.Collections.Generic;

namespace CnnFromScratch.Training
{
    public class SGDOptimizer : IOptimizer
    {
        private readonly float _momentum;
        private readonly float _weightDecay;
        private readonly float _clipValue;
        private readonly Dictionary<ILayer, Velocity> _velocities = new();

        public SGDOptimizer(float momentum = 0.9f, float weightDecay = 1e-4f, float clipValue = 5.0f)
        {
            _momentum = momentum;
            _weightDecay = weightDecay;
            _clipValue = clipValue;
        }

        public void UpdateLayer(ILayer layer, float learningRate)
        {
            // Skip layers without parameters
            if (layer is ReLULayer || layer is MaxPoolLayer || layer is SoftmaxLayer )
                return;

            if (layer is Conv2DLayer conv)
            {
                UpdateConvLayer(conv, learningRate);
            }
            else if (layer is DenseLayer dense)
            {
                UpdateDenseLayer(dense, learningRate);
            }
            else if (layer is BatchNormLayer batchNorm)
            {
                UpdateBatchNormLayer(batchNorm, learningRate);
            }
        }

        private void UpdateConvLayer(Conv2DLayer layer, float learningRate)
        {
            var weights = layer.GetWeights();
            var biases = layer.GetBiases();
            var weightGrads = layer.WeightGradients;
            var biasGrads = layer.BiasGradients;

            if (!_velocities.TryGetValue(layer, out var velocity))
            {
                velocity = new Velocity
                {
                    ConvWeights = new float[weightGrads.GetLength(0), weightGrads.GetLength(1), weightGrads.GetLength(2), weightGrads.GetLength(3)],
                    ConvBiases = new float[biasGrads.Length]
                };
                _velocities[layer] = velocity;
            }

            var vWeights = velocity.ConvWeights;
            var vBiases = velocity.ConvBiases;

            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                {
                    for (int k = 0; k < weights.GetLength(2); k++)
                    {
                        for (int l = 0; l < weights.GetLength(3); l++)
                        {
                            float grad = Clip(weightGrads[i, j, k, l]);
                            vWeights[i, j, k, l] = _momentum * vWeights[i, j, k, l] - learningRate * (grad + _weightDecay * weights[i, j, k, l]);
                            weights[i, j, k, l] += vWeights[i, j, k, l];
                        }
                    }
                }
            }

            for (int i = 0; i < biases.Length; i++)
            {
                float grad = Clip(biasGrads[i]);
                vBiases[i] = _momentum * vBiases[i] - learningRate * grad;
                biases[i] += vBiases[i];
            }

            layer.SetWeightsAndBiases(weights, biases);
        }
        private void UpdateBatchNormLayer(BatchNormLayer layer, float learningRate)
        {
            var gamma = layer.GetWeights();
            var beta = layer.GetBiases();
            var gammaGrads = layer.WeightGradients;
            var betaGrads = layer.BiasGradients;

            if (!_velocities.TryGetValue(layer, out var velocity))
            {
                velocity = new Velocity
                {
                    BatchNormWeights = new float[gamma.Length],
                    BatchNormBiases = new float[beta.Length]
                };
                _velocities[layer] = velocity;
            }

            var vWeights = velocity.BatchNormWeights!;
            var vBiases = velocity.BatchNormBiases!;

            for (int i = 0; i < gamma.Length; i++)
            {
                float grad = Clip(gammaGrads[i]);
                vWeights[i] = _momentum * vWeights[i] - learningRate * (grad + _weightDecay * gamma[i]);
                gamma[i] += vWeights[i];
            }

            for (int i = 0; i < beta.Length; i++)
            {
                float grad = Clip(betaGrads[i]);
                vBiases[i] = _momentum * vBiases[i] - learningRate * grad;
                beta[i] += vBiases[i];
            }

            layer.SetWeightsAndBiases(gamma, beta);
        }

        private void UpdateDenseLayer(DenseLayer layer, float learningRate)
        {
            var weights = layer.GetWeights();
            var biases = layer.GetBiases();
            var weightGrads = layer.WeightGradients;
            var biasGrads = layer.BiasGradients;

            if (!_velocities.TryGetValue(layer, out var velocity))
            {
                velocity = new Velocity
                {
                    DenseWeights = new Matrix(weights.Rows, weights.Cols),
                    DenseBiases = new float[biases.Length]
                };
                _velocities[layer] = velocity;
            }

            var vWeights = velocity.DenseWeights;
            var vBiases = velocity.DenseBiases;

            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Cols; j++)
                {
                    float grad = Clip(weightGrads[i, j]);
                    vWeights[i, j] = _momentum * vWeights[i, j] - learningRate * (grad + _weightDecay * weights[i, j]);
                    weights[i, j] += vWeights[i, j];
                }
            }

            for (int i = 0; i < biases.Length; i++)
            {
                float grad = Clip(biasGrads[i]);
                vBiases[i] = _momentum * vBiases[i] - learningRate * grad;
                biases[i] += vBiases[i];
            }

            layer.SetWeightsAndBiases(weights, biases);
        }

        private float Clip(float value)
        {
            return Math.Clamp(value, -_clipValue, _clipValue);
        }

        private class Velocity
        {
            // For Conv2D layers
            public float[,,,]? ConvWeights;
            public float[]? ConvBiases;

            // For Dense layers
            public Matrix? DenseWeights;
            public float[]? DenseBiases;

            public float[] BatchNormWeights { get; internal set; }
            public float[] BatchNormBiases { get; internal set; }
        }
    }
}
