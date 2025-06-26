using CnnFromScratch.Core;
using System;

namespace CnnFromScratch.Layers
{
    public class Conv2DLayer : ILayer
    {
        public int InputChannels { get; }
        public int OutputChannels { get; }
        public int KernelSize { get; }
        public int Stride { get; }
        public int Padding { get; }

        private float[,,,] Weights; // [Out, In, K, K]
        private float[] Biases;
        private Tensor3D? _lastInput;
        private Tensor3D? _lastPaddedInput;

        // Gradients for optimization
        public float[,,,] WeightGradients { get; private set; }
        public float[] BiasGradients { get; private set; }

        private Random random = new Random();

        public Conv2DLayer(int inputChannels, int outputChannels, int kernelSize, int stride = 1, int padding = 0)
        {
            InputChannels = inputChannels;
            OutputChannels = outputChannels;
            KernelSize = kernelSize;
            Stride = stride;
            Padding = padding;

            Weights = new float[outputChannels, inputChannels, kernelSize, kernelSize];
            Biases = new float[outputChannels];
            WeightGradients = new float[outputChannels, inputChannels, kernelSize, kernelSize];
            BiasGradients = new float[outputChannels];

            InitWeights();
        }

        private void InitWeights()
        {
            //// Xavier/Glorot initialization
            //float stddev = (float)Math.Sqrt(2.0 / (InputChannels * KernelSize * KernelSize + OutputChannels));
            // He initialization (more common for ReLU activations) 
            float stddev = (float)Math.Sqrt(2.0 / (InputChannels * KernelSize * KernelSize));

            for (int o = 0; o < OutputChannels; o++)
            {
                for (int i = 0; i < InputChannels; i++)
                    for (int y = 0; y < KernelSize; y++)
                        for (int x = 0; x < KernelSize; x++)
                            Weights[o, i, y, x] = (float)(random.NextDouble() * 2 - 1) * stddev;

                Biases[o] = 0; // Initialize biases to zero
            }
        }

        private Tensor3D Pad(Tensor3D input)
        {
            if (Padding == 0)
                return input;

            var padded = new Tensor3D(input.Channels, input.Height + 2 * Padding, input.Width + 2 * Padding);
            for (int c = 0; c < input.Channels; c++)
                for (int h = 0; h < input.Height; h++)
                    for (int w = 0; w < input.Width; w++)
                        padded[c, h + Padding, w + Padding] = input[c, h, w];

            return padded;
        }

        public Tensor3D Forward(Tensor3D input)
        {
            if (input.Channels != InputChannels)
                throw new ArgumentException("Input channels do not match layer configuration.");

            _lastInput = input.Clone();
            _lastPaddedInput = Pad(input);

            int H_out = (_lastPaddedInput.Height - KernelSize) / Stride + 1;
            int W_out = (_lastPaddedInput.Width - KernelSize) / Stride + 1;

            var output = new Tensor3D(OutputChannels, H_out, W_out);

            for (int oc = 0; oc < OutputChannels; oc++)
            {
                for (int oh = 0; oh < H_out; oh++)
                {
                    for (int ow = 0; ow < W_out; ow++)
                    {
                        float sum = 0f;

                        for (int ic = 0; ic < InputChannels; ic++)
                        {
                            for (int kh = 0; kh < KernelSize; kh++)
                            {
                                for (int kw = 0; kw < KernelSize; kw++)
                                {
                                    int ih = oh * Stride + kh;
                                    int iw = ow * Stride + kw;

                                    sum += _lastPaddedInput[ic, ih, iw] * Weights[oc, ic, kh, kw];
                                }
                            }
                        }

                        sum += Biases[oc];
                        output[oc, oh, ow] = sum;
                    }
                }
            }

            return output;
        }

        public Tensor3D Backward(Tensor3D outputGradient)
        {
            if (_lastInput == null || _lastPaddedInput == null)
                throw new InvalidOperationException("Backward called before Forward");

            // Initialize gradients
            Array.Clear(WeightGradients, 0, WeightGradients.Length);
            Array.Clear(BiasGradients, 0, BiasGradients.Length);

            // Initialize input gradients
            var inputGradient = new Tensor3D(InputChannels, _lastInput.Height, _lastInput.Width);
            var paddedInputGradient = new Tensor3D(InputChannels, _lastPaddedInput.Height, _lastPaddedInput.Width);

            // Compute gradients for weights and biases
            for (int oc = 0; oc < OutputChannels; oc++)
            {
                for (int oh = 0; oh < outputGradient.Height; oh++)
                {
                    for (int ow = 0; ow < outputGradient.Width; ow++)
                    {
                        float gradOutput = outputGradient[oc, oh, ow];
                        BiasGradients[oc] += gradOutput;

                        for (int ic = 0; ic < InputChannels; ic++)
                        {
                            for (int kh = 0; kh < KernelSize; kh++)
                            {
                                for (int kw = 0; kw < KernelSize; kw++)
                                {
                                    int ih = oh * Stride + kh;
                                    int iw = ow * Stride + kw;

                                    // Gradient w.r.t weights
                                    WeightGradients[oc, ic, kh, kw] += _lastPaddedInput[ic, ih, iw] * gradOutput;

                                    // Gradient w.r.t input
                                    paddedInputGradient[ic, ih, iw] += Weights[oc, ic, kh, kw] * gradOutput;
                                }
                            }
                        }
                    }
                }
            }

            // Remove padding from input gradients
            for (int ic = 0; ic < InputChannels; ic++)
                for (int h = 0; h < _lastInput.Height; h++)
                    for (int w = 0; w < _lastInput.Width; w++)
                        inputGradient[ic, h, w] = paddedInputGradient[ic, h + Padding, w + Padding];

            return inputGradient;
        }

        public void SetWeightsAndBiases(float[,,,] weights, float[] biases)
        {
            if (weights.GetLength(0) != OutputChannels ||
                weights.GetLength(1) != InputChannels ||
                weights.GetLength(2) != KernelSize ||
                weights.GetLength(3) != KernelSize)
                throw new ArgumentException("Weight dimensions must match layer configuration");
            
            if (biases.Length != OutputChannels)
                throw new ArgumentException("Bias array length must match output channels");

            Weights = weights;
            Biases = biases;
        }

        public float[,,,] GetWeights()
        {
            return Weights;
        }

        public float[] GetBiases()
        {
            return Biases;
        }

        object ILayer.GetWeights()
        {
            return GetWeights();
        }

        public void SetWeightsAndBiases(object weights, float[] biases)
        {
            if (weights is float[,,,] convWeights)
            {
                SetWeightsAndBiases(convWeights, biases);
            }
            else
            {
                throw new ArgumentException($"Weights must be of type float[,,,] for Conv2DLayer, but got {weights?.GetType().Name ?? "null"}");
            }
        }
    }
}
