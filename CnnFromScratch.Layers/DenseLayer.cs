using CnnFromScratch.Core;
using System;

namespace CnnFromScratch.Layers
{
    public class DenseLayer : ILayer
    {
        public int InputSize { get; }
        public int OutputSize { get; }
        
        private Matrix _weights;
        private float[] _biases;
        private Tensor3D? _lastInput;
        
        // Gradients for optimization
        public Matrix WeightGradients { get; private set; }
        public float[] BiasGradients { get; private set; }

        public DenseLayer(int inputSize, int outputSize)
        {
            if (inputSize <= 0) throw new ArgumentException("Input size must be positive", nameof(inputSize));
            if (outputSize <= 0) throw new ArgumentException("Output size must be positive", nameof(outputSize));

            InputSize = inputSize;
            OutputSize = outputSize;

            // Initialize weights and biases
            _weights = new Matrix(outputSize, inputSize, fillZero: false);
            _biases = new float[outputSize];
            WeightGradients = new Matrix(outputSize, inputSize);
            BiasGradients = new float[outputSize];

            InitializeWeights();
        }

        private void InitializeWeights()
        {
            // He initialization
            float stddev = (float)Math.Sqrt(2.0 / InputSize);
            var random = new Random();
            
            for (int i = 0; i < OutputSize; i++)
            {
                for (int j = 0; j < InputSize; j++)
                {
                    _weights[i, j] = (float)(random.NextDouble() * 2 - 1) * stddev;
                }
                _biases[i] = 0; // Initialize biases to zero
            }
        }

        private float[] FlattenInput(Tensor3D input)
        {
            int flatSize = input.Channels * input.Height * input.Width;
            var flattened = new float[flatSize];
            int idx = 0;

            for (int c = 0; c < input.Channels; c++)
                for (int h = 0; h < input.Height; h++)
                    for (int w = 0; w < input.Width; w++)
                        flattened[idx++] = input[c, h, w];

            return flattened;
        }

        private Tensor3D ReshapeToTensor3D(float[] array)
        {
            var output = new Tensor3D(1, 1, array.Length);
            for (int i = 0; i < array.Length; i++)
            {
                output[0, 0, i] = array[i];
            }
            return output;
        }

        public Tensor3D Forward(Tensor3D input)
        {
            _lastInput = input.Clone();
            
            // Flatten input
            float[] flatInput = FlattenInput(input);
            
            if (flatInput.Length != InputSize)
                throw new ArgumentException($"Input size mismatch. Expected {InputSize}, got {flatInput.Length}");

            // Compute output: y = Wx + b
            var output = new float[OutputSize];
            for (int i = 0; i < OutputSize; i++)
            {
                float sum = _biases[i];
                for (int j = 0; j < InputSize; j++)
                {
                    sum += flatInput[j] * _weights[i, j];
                }
                output[i] = sum;
            }

            return ReshapeToTensor3D(output);
        }

        public Tensor3D Backward(Tensor3D outputGradient)
        {
            if (_lastInput == null)
                throw new InvalidOperationException("Backward called before Forward");

            float[] flatInput = FlattenInput(_lastInput);
            float[] flatGradient = FlattenInput(outputGradient);

            // Compute gradients with respect to weights and biases
            for (int i = 0; i < OutputSize; i++)
            {
                BiasGradients[i] = flatGradient[i];
                for (int j = 0; j < InputSize; j++)
                {
                    WeightGradients[i, j] = flatGradient[i] * flatInput[j];
                }
            }

            // Compute gradient with respect to input
            var inputGradient = new float[InputSize];
            for (int j = 0; j < InputSize; j++)
            {
                float sum = 0;
                for (int i = 0; i < OutputSize; i++)
                {
                    sum += flatGradient[i] * _weights[i, j];
                }
                inputGradient[j] = sum;
            }

            // Reshape input gradient to match input dimensions
            return ReshapeToTensor3D(inputGradient);
        }

        public void SetWeightsAndBiases(Matrix weights, float[] biases)
        {
            if (weights.Rows != OutputSize || weights.Cols != InputSize)
                throw new ArgumentException("Weight matrix dimensions must match layer configuration");
            if (biases.Length != OutputSize)
                throw new ArgumentException("Bias array length must match output size");

            _weights = weights;
            _biases = biases;
        }

        public Matrix GetWeights()
        {
            return _weights.Clone();
        }

        public float[] GetBiases()
        {
            return (float[])_biases.Clone();
        }

        object ILayer.GetWeights()
        {
            return GetWeights();
        }

        public void SetWeightsAndBiases(object weights, float[] biases)
        {
            if (weights is Matrix matrix)
            {
                SetWeightsAndBiases(matrix, biases);
            }
            else
            {
                throw new ArgumentException($"Weights must be of type Matrix for DenseLayer, but got {weights?.GetType().Name ?? "null"}");
            }
        }
    }
}