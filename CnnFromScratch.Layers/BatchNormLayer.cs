using CnnFromScratch.Core;
using System;

namespace CnnFromScratch.Layers
{
    public class BatchNormLayer : ILayer
    {
        private readonly float _momentum;
        private readonly float _epsilon;
        private float[] _gamma;
        private float[] _beta;
        private float[] _gammaGradients;
        private float[] _betaGradients;
        private float[] _runningMean;
        private float[] _runningVariance;
        private float[] _lastMean;
        private float[] _lastVariance;
        private float[] _lastStd;
        private Tensor4D _lastInput;
        private Tensor4D _lastNormalized;
        private int _channels;
        private bool _isTraining;

        public BatchNormLayer(float momentum = 0.9f, float epsilon = 1e-5f)
        {
            _momentum = momentum;
            _epsilon = epsilon;
            _isTraining = true;
        }

        public void SetTrainingMode(bool isTraining)
        {
            _isTraining = isTraining;
        }

        public Tensor4D Forward(Tensor4D input)
        {
            if (_gamma == null)
            {
                _channels = input.Channels;
                _gamma = new float[_channels];
                _beta = new float[_channels];
                _gammaGradients = new float[_channels];
                _betaGradients = new float[_channels];
                _runningMean = new float[_channels];
                _runningVariance = new float[_channels];
                for (int c = 0; c < _channels; c++)
                {
                    _gamma[c] = 1f;
                    _beta[c] = 0f;
                }
            }

            int batchSize = input.BatchSize;
            int height = input.Height;
            int width = input.Width;
            int N = batchSize * height * width;

            var output = new Tensor4D(batchSize, _channels, height, width);
            _lastInput = input;
            _lastNormalized = new Tensor4D(batchSize, _channels, height, width);
            _lastMean = new float[_channels];
            _lastVariance = new float[_channels];
            _lastStd = new float[_channels];

            for (int c = 0; c < _channels; c++)
            {
                // Compute mean over batch, height, width for each channel
                float mean = 0f;
                for (int b = 0; b < batchSize; b++)
                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                            mean += input[b, c, h, w];
                mean /= N;
                _lastMean[c] = mean;

                // Compute variance
                float variance = 0f;
                for (int b = 0; b < batchSize; b++)
                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                        {
                            float val = input[b, c, h, w] - mean;
                            variance += val * val;
                        }
                variance /= N;
                _lastVariance[c] = variance;

                float std = (float)Math.Sqrt(variance + _epsilon);
                if (float.IsNaN(std) || float.IsInfinity(std)) std = 1f;
                _lastStd[c] = std;

                if (_isTraining)
                {
                    _runningMean[c] = _momentum * _runningMean[c] + (1 - _momentum) * mean;
                    _runningVariance[c] = _momentum * _runningVariance[c] + (1 - _momentum) * variance;
                }
                else
                {
                    mean = _runningMean[c];
                    std = (float)Math.Sqrt(_runningVariance[c] + _epsilon);
                }

                // Normalize, scale, shift
                for (int b = 0; b < batchSize; b++)
                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                        {
                            float x = input[b, c, h, w];
                            float xHat = (x - mean) / std;
                            _lastNormalized[b, c, h, w] = xHat;
                            output[b, c, h, w] = _gamma[c] * xHat + _beta[c];
                        }
            }

            return output;
        }

        public Tensor4D Backward(Tensor4D outputGradient)
        {
            int batchSize = _lastInput.BatchSize;
            int height = _lastInput.Height;
            int width = _lastInput.Width;
            int N = batchSize * height * width;

            var inputGradient = new Tensor4D(batchSize, _channels, height, width);

            Array.Clear(_gammaGradients, 0, _gammaGradients.Length);
            Array.Clear(_betaGradients, 0, _betaGradients.Length);

            for (int c = 0; c < _channels; c++)
            {
                float gamma = _gamma[c];
                float std = _lastStd[c];

                float sumDOut = 0f;
                float sumDOutXHat = 0f;

                // Gradients for gamma/beta
                for (int b = 0; b < batchSize; b++)
                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                        {
                            float dOut = outputGradient[b, c, h, w];
                            float xHat = _lastNormalized[b, c, h, w];
                            _gammaGradients[c] += dOut * xHat;
                            _betaGradients[c] += dOut;
                            sumDOut += dOut;
                            sumDOutXHat += dOut * xHat;
                        }

                // Input gradient calculation
                for (int b = 0; b < batchSize; b++)
                    for (int h = 0; h < height; h++)
                        for (int w = 0; w < width; w++)
                        {
                            float xHat = _lastNormalized[b, c, h, w];
                            float dOut = outputGradient[b, c, h, w];
                            float dx = (1f / N) * gamma / std *
                                       (N * dOut - sumDOut - xHat * sumDOutXHat);
                            inputGradient[b, c, h, w] = dx;
                        }
            }

            return inputGradient;
        }

        public float[] GetWeights() => _gamma;
        public float[] GetBiases() => _beta;

        public void SetWeightsAndBiases(float[] gamma, float[] beta)
        {
            if (gamma.Length != _channels || beta.Length != _channels)
                throw new ArgumentException("Gamma and beta must match channel count");

            _gamma = gamma;
            _beta = beta;
        }

        public void SetWeightsAndBiases(object weights, float[] biases)
        {
            if (weights is float[] gamma)
                SetWeightsAndBiases(gamma, biases);
            else
                throw new ArgumentException("Weights must be float[] for BatchNormLayer");
        }

        object ILayer.GetWeights() => GetWeights();
        public float[] GetGammaGradients() => _gammaGradients;
        public float[] GetBetaGradients() => _betaGradients;
        public float[] WeightGradients => _gammaGradients;
        public float[] BiasGradients => _betaGradients;

        public Tensor3D Forward(Tensor3D input)
        {
            throw new NotImplementedException();
        }

        public Tensor3D Backward(Tensor3D outputGradient)
        {
            throw new NotImplementedException();
        }
    }
}
