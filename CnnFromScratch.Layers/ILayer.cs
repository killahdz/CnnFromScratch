using CnnFromScratch.Core;

namespace CnnFromScratch.Layers
{
    public interface ILayer
    {
        // Forward pass on a single sample (Tensor3D)
        Tensor3D Forward(Tensor3D input);

        // Backward pass on single sample gradient
        Tensor3D Backward(Tensor3D outputGradient);

        // Serialization support
        object GetWeights();
        float[] GetBiases();
        void SetWeightsAndBiases(object weights, float[] biases);

        // Batch forward - default implementation uses single-sample Forward
        Tensor4D Forward(Tensor4D inputBatch)
        {
            int batchSize = inputBatch.BatchSize;

            // Get output shape by forwarding first sample
            var firstOutput = Forward(inputBatch.GetSlice(0));
            int channels = firstOutput.Channels;
            int height = firstOutput.Height;
            int width = firstOutput.Width;

            var outputs = new Tensor4D(batchSize, channels, height, width);
            outputs.SetSlice(0, firstOutput);

            for (int i = 1; i < batchSize; i++)
            {
                var singleInput = inputBatch.GetSlice(i);
                var singleOutput = Forward(singleInput);
                outputs.SetSlice(i, singleOutput);
            }
            return outputs;
        }

        // Batch backward - default implementation uses single-sample Backward
        Tensor4D Backward(Tensor4D gradientBatch)
        {
            int batchSize = gradientBatch.BatchSize;

            // Get input gradient shape by backwarding first sample
            var firstInputGrad = Backward(gradientBatch.GetSlice(0));
            int channels = firstInputGrad.Channels;
            int height = firstInputGrad.Height;
            int width = firstInputGrad.Width;

            var inputGradients = new Tensor4D(batchSize, channels, height, width);
            inputGradients.SetSlice(0, firstInputGrad);

            for (int i = 1; i < batchSize; i++)
            {
                var singleGrad = gradientBatch.GetSlice(i);
                var inputGrad = Backward(singleGrad);
                inputGradients.SetSlice(i, inputGrad);
            }
            return inputGradients;
        }
    }
}
