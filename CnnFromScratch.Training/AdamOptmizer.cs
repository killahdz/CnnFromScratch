using System;
using System.Collections.Generic;
using CnnFromScratch.Core;
using CnnFromScratch.Layers;

namespace CnnFromScratch.Training
{
    public class AdamOptimizer : IOptimizer
    {
        private readonly float _beta1;
        private readonly float _beta2;
        private readonly float _epsilon;
        private readonly Dictionary<ILayer, int> _stepCounts = new();
        private readonly Dictionary<ILayer, object> _momentEstimates = new();

        public AdamOptimizer(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
        }

        public void UpdateLayer(ILayer layer, float learningRate)
        {
            if (layer is ReLULayer || layer is MaxPoolLayer || layer is SoftmaxLayer)
                return;

            if (!_stepCounts.ContainsKey(layer))
                _stepCounts[layer] = 0;

            _stepCounts[layer]++;

            if (layer is Conv2DLayer conv)
                UpdateConvLayer(conv, learningRate, _stepCounts[layer]);
            else if (layer is DenseLayer dense)
                UpdateDenseLayer(dense, learningRate, _stepCounts[layer]);
            else if (layer is BatchNormLayer batchNorm)
                UpdateBatchNormLayer(batchNorm, learningRate, _stepCounts[layer]);
        }

        private void UpdateConvLayer(Conv2DLayer layer, float learningRate, int t)
        {
            var weights = layer.GetWeights();
            var biases = layer.GetBiases();
            var weightGrads = layer.WeightGradients;
            var biasGrads = layer.BiasGradients;

            if (!_momentEstimates.ContainsKey(layer))
            {
                _momentEstimates[layer] = new
                {
                    MWeights = new float[weightGrads.GetLength(0), weightGrads.GetLength(1),
                                        weightGrads.GetLength(2), weightGrads.GetLength(3)],
                    VWeights = new float[weightGrads.GetLength(0), weightGrads.GetLength(1),
                                        weightGrads.GetLength(2), weightGrads.GetLength(3)],
                    MBiases = new float[biasGrads.Length],
                    VBiases = new float[biasGrads.Length]
                };
            }

            var moments = _momentEstimates[layer];
            var mWeights = ((dynamic)moments).MWeights as float[,,,];
            var vWeights = ((dynamic)moments).VWeights as float[,,,];
            var mBiases = ((dynamic)moments).MBiases as float[];
            var vBiases = ((dynamic)moments).VBiases as float[];

            float biasCorrection1 = 1 - (float)Math.Pow(_beta1, t);
            float biasCorrection2 = 1 - (float)Math.Pow(_beta2, t);

            int w0 = weights.GetLength(0);
            int w1 = weights.GetLength(1);
            int w2 = weights.GetLength(2);
            int w3 = weights.GetLength(3);

            for (int i = 0; i < w0; i++)
            {
                for (int j = 0; j < w1; j++)
                {
                    for (int k = 0; k < w2; k++)
                    {
                        for (int l = 0; l < w3; l++)
                        {
                            float grad = weightGrads[i, j, k, l];
                            mWeights[i, j, k, l] = _beta1 * mWeights[i, j, k, l] + (1 - _beta1) * grad;
                            vWeights[i, j, k, l] = _beta2 * vWeights[i, j, k, l] + (1 - _beta2) * (grad * grad);
                            float mHat = mWeights[i, j, k, l] / biasCorrection1;
                            float vHat = vWeights[i, j, k, l] / biasCorrection2;
                            weights[i, j, k, l] -= learningRate * mHat / (float)(Math.Sqrt(vHat) + _epsilon);
                        }
                    }
                }
            }

            for (int i = 0; i < biases.Length; i++)
            {
                float grad = biasGrads[i];
                mBiases[i] = _beta1 * mBiases[i] + (1 - _beta1) * grad;
                vBiases[i] = _beta2 * vBiases[i] + (1 - _beta2) * (grad * grad);
                float mHat = mBiases[i] / biasCorrection1;
                float vHat = vBiases[i] / biasCorrection2;
                biases[i] -= learningRate * mHat / (float)(Math.Sqrt(vHat) + _epsilon);
            }

            layer.SetWeightsAndBiases(weights, biases);
        }

        private void UpdateDenseLayer(DenseLayer layer, float learningRate, int t)
        {
            var weights = layer.GetWeights();
            var biases = layer.GetBiases();
            var weightGrads = layer.WeightGradients;
            var biasGrads = layer.BiasGradients;

            if (!_momentEstimates.ContainsKey(layer))
            {
                _momentEstimates[layer] = new
                {
                    MWeights = new Matrix(weights.Rows, weights.Cols),
                    VWeights = new Matrix(weights.Rows, weights.Cols),
                    MBiases = new float[biases.Length],
                    VBiases = new float[biases.Length]
                };
            }

            var moments = _momentEstimates[layer];
            var mWeights = ((dynamic)moments).MWeights as Matrix;
            var vWeights = ((dynamic)moments).VWeights as Matrix;
            var mBiases = ((dynamic)moments).MBiases as float[];
            var vBiases = ((dynamic)moments).VBiases as float[];

            float biasCorrection1 = 1 - (float)Math.Pow(_beta1, t);
            float biasCorrection2 = 1 - (float)Math.Pow(_beta2, t);

            for (int i = 0; i < weights.Rows; i++)
            {
                for (int j = 0; j < weights.Cols; j++)
                {
                    float grad = weightGrads[i, j];
                    mWeights[i, j] = _beta1 * mWeights[i, j] + (1 - _beta1) * grad;
                    vWeights[i, j] = _beta2 * vWeights[i, j] + (1 - _beta2) * (grad * grad);
                    float mHat = mWeights[i, j] / biasCorrection1;
                    float vHat = vWeights[i, j] / biasCorrection2;
                    weights[i, j] -= learningRate * mHat / (float)(Math.Sqrt(vHat) + _epsilon);
                }
            }

            for (int i = 0; i < biases.Length; i++)
            {
                float grad = biasGrads[i];
                mBiases[i] = _beta1 * mBiases[i] + (1 - _beta1) * grad;
                vBiases[i] = _beta2 * vBiases[i] + (1 - _beta2) * (grad * grad);
                float mHat = mBiases[i] / biasCorrection1;
                float vHat = vBiases[i] / biasCorrection2;
                biases[i] -= learningRate * mHat / (float)(Math.Sqrt(vHat) + _epsilon);
            }

            layer.SetWeightsAndBiases(weights, biases);
        }

        private void UpdateBatchNormLayer(BatchNormLayer layer, float learningRate, int t)
        {
            var gamma = layer.GetWeights();
            var beta = layer.GetBiases();
            var gammaGrads = layer.WeightGradients;
            var betaGrads = layer.BiasGradients;

            if (!_momentEstimates.ContainsKey(layer))
            {
                _momentEstimates[layer] = new
                {
                    MWeights = new float[gamma.Length],
                    VWeights = new float[gamma.Length],
                    MBiases = new float[beta.Length],
                    VBiases = new float[beta.Length]
                };
            }

            var moments = _momentEstimates[layer];
            var mWeights = ((dynamic)moments).MWeights as float[];
            var vWeights = ((dynamic)moments).VWeights as float[];
            var mBiases = ((dynamic)moments).MBiases as float[];
            var vBiases = ((dynamic)moments).VBiases as float[];

            float biasCorrection1 = 1 - (float)Math.Pow(_beta1, t);
            float biasCorrection2 = 1 - (float)Math.Pow(_beta2, t);

            for (int i = 0; i < gamma.Length; i++)
            {
                float grad = gammaGrads[i];
                mWeights[i] = _beta1 * mWeights[i] + (1 - _beta1) * grad;
                vWeights[i] = _beta2 * vWeights[i] + (1 - _beta2) * (grad * grad);
                float mHat = mWeights[i] / biasCorrection1;
                float vHat = vWeights[i] / biasCorrection2;
                gamma[i] -= learningRate * mHat / (float)(Math.Sqrt(vHat) + _epsilon);
            }

            for (int i = 0; i < beta.Length; i++)
            {
                float grad = betaGrads[i];
                mBiases[i] = _beta1 * mBiases[i] + (1 - _beta1) * grad;
                vBiases[i] = _beta2 * vBiases[i] + (1 - _beta2) * (grad * grad);
                float mHat = mBiases[i] / biasCorrection1;
                float vHat = vBiases[i] / biasCorrection2;
                beta[i] -= learningRate * mHat / (float)(Math.Sqrt(vHat) + _epsilon);
            }

            layer.SetWeightsAndBiases(gamma, beta);
        }
    }
}