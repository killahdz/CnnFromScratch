using CnnFromScratch.Core;

namespace CnnFromScratch.Layers
{
    public class GlobalAveragePoolLayer : ILayer
    {
        private int inputChannels;

        public Tensor3D Forward(Tensor3D input)
        {
            inputChannels = input.Channels;
            int height = input.Height;
            int width = input.Width;

            var output = new Tensor3D(inputChannels, 1, 1);

            for (int c = 0; c < inputChannels; c++)
            {
                float sum = 0f;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        sum += input[c, h, w];
                    }
                }
                output[c, 0, 0] = sum / (height * width);
            }

            return output;
        }

        public Tensor3D Backward(Tensor3D outputGradient)
        {
            // Gradient is spread equally across all spatial positions in each channel
            var grad = new Tensor3D(inputChannels, 1, 1);
            float[,,] gradInput = new float[inputChannels, Height, Width];

            float scale = 1.0f / (Height * Width);
            var inputGrad = new Tensor3D(inputChannels, Height, Width);

            for (int c = 0; c < inputChannels; c++)
            {
                float gradVal = outputGradient[c, 0, 0] * scale;
                for (int h = 0; h < Height; h++)
                {
                    for (int w = 0; w < Width; w++)
                    {
                        inputGrad[c, h, w] = gradVal;
                    }
                }
            }

            return inputGrad;
        }

        public object GetWeights() => null;
        public float[] GetBiases() => null;
        public void SetWeightsAndBiases(object weights, float[] biases) { }

        // Not strictly required, but helpful if shape tracking is manual
        private int Height, Width;
        public Tensor3D ForwardWithShapeTracking(Tensor3D input)
        {
            Height = input.Height;
            Width = input.Width;
            return Forward(input);
        }
    }
}
