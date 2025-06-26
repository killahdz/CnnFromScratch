using CnnFromScratch.Core;
using System;

namespace CnnFromScratch.Layers
{
    public class ReLULayer : ILayer
    {
        private Tensor3D? _lastInput;

        public Tensor3D Forward(Tensor3D input)
        {
            _lastInput = input.Clone();
            var output = input.Clone();
            for (int c = 0; c < input.Channels; c++)
                for (int h = 0; h < input.Height; h++)
                    for (int w = 0; w < input.Width; w++)
                        output[c, h, w] = Math.Max(0, input[c, h, w]);
            return output;
        }

        public Tensor3D Backward(Tensor3D outputGradient)
        {
            if (_lastInput == null)
                throw new InvalidOperationException("Backward called before Forward");

            var inputGradient = outputGradient.Clone();

            // ReLU derivative: 1 if input was > 0, 0 otherwise
            for (int c = 0; c < outputGradient.Channels; c++)
                for (int h = 0; h < outputGradient.Height; h++)
                    for (int w = 0; w < outputGradient.Width; w++)
                        inputGradient[c, h, w] *= _lastInput[c, h, w] > 0 ? 1 : 0;

            return inputGradient;
        }

        public object GetWeights()
        {
            return Array.Empty<float>();
        }

        public float[] GetBiases()
        {
            return Array.Empty<float>();
        }

        public void SetWeightsAndBiases(object weights, float[] biases)
        {
            // ReLU has no weights or biases, so nothing to set
        }
    }
}

