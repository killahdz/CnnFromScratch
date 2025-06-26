//using CnnFromScratch.Core;
//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Net.NetworkInformation;
//using System.Text;
//using System.Threading.Tasks;


//namespace CnnFromScratch.Layers
//    {
//        public class DropoutLayer : ILayer
//        {
//            private readonly float dropoutRate; 
//        private readonly Random random; 
//        private bool isTraining; 
//        private Tensor3D mask; // Stores dropout mask for single-sample backpropagation`

//                /// <summary>
//    /// Initializes a DropoutLayer with the specified dropout rate.
//    /// </summary>
//    /// <param name="dropoutRate">Probability of dropping a unit (0 to 1).</param>
//    public DropoutLayer(float dropoutRate = 0.5f)
//    {
//        if (dropoutRate < 0 || dropoutRate >= 1)
//            throw new ArgumentException("Dropout rate must be between 0 and 1.", nameof(dropoutRate));

//        this.dropoutRate = dropoutRate;
//        random = new Random();
//        isTraining = true;
//    }

//    /// <summary>
//    /// Gets or sets whether the layer is in training mode.
//    /// </summary>
//    public bool IsTraining
//    {
//        get => isTraining;
//        set => isTraining = value;
//    }

//    /// <summary>
//    /// Performs the forward pass on a single sample, applying dropout during training.
//    /// </summary>
//    /// <param name="input">Input Tensor3D.</param>
//    /// <returns>Output Tensor3D with dropout applied (if training).</returns>
//    public Tensor3D Forward(Tensor3D input)
//    {
//        if (!isTraining)
//        {
//            // During inference, return a clone of the input
//            return input.Clone();
//        }

//        // Generate dropout mask and apply scaling
//        int channels = input.Channels;
//        int height = input.Height;
//        int width = input.Width;
//        float scale = 1f / (1f - dropoutRate);
//        Tensor3D output = new Tensor3D(channels, height, width);
//        mask = new Tensor3D(channels, height, width);

//        for (int c = 0; c < channels; c++)
//        {
//            for (int h = 0; h < height; h++)
//            {
//                for (int w = 0; w < width; w++)
//                {
//                    float maskValue = random.NextDouble() > dropoutRate ? 1f : 0f;
//                    mask[c, h, w] = maskValue;
//                    output[c, h, w] = input[c, h, w] * maskValue * scale;
//                }
//            }
//        }

//        return output;
//    }

//    /// <summary>
//    /// Performs the backward pass on a single sample gradient.
//    /// </summary>
//    /// <param name="outputGradient">Gradient of the loss w.r.t. output.</param>
//    /// <returns>Gradient of the loss w.r.t. input.</returns>
//    public Tensor3D Backward(Tensor3D outputGradient)
//    {
//        if (!isTraining)
//        {
//            // During inference, return a clone of the gradient
//            return outputGradient.Clone();
//        }

//        int channels = outputGradient.Channels;
//        int height = outputGradient.Height;
//        int width = outputGradient.Width;
//        float scale = 1f / (1f - dropoutRate);
//        Tensor3D inputGradient = new Tensor3D(channels, height, width);

//        // Apply dropout mask to gradients
//        for (int c = 0; c < channels; c++)
//        {
//            for (int h = 0; h < height; h++)
//            {
//                for (int w = 0; w < width; w++)
//                {
//                    inputGradient[c, h, w] = outputGradient[c, h, w] * mask[c, h, w] * scale;
//                }
//            }
//        }

//        return inputGradient;
//    }

//    /// <summary>
//    /// Returns null as DropoutLayer has no weights.
//    /// </summary>
//    public object GetWeights() => null;

//    /// <summary>
//    /// Returns null as DropoutLayer has no biases.
//    /// </summary>
//    public float[] GetBiases() => null;

//    /// <summary>
//    /// No-op as DropoutLayer has no weights or biases.
//    /// </summary>
//    public void SetWeightsAndBiases(object weights, float[] biases)
//    {
//        // No weights or biases to set
//    }
//}
//    {
        
//    }
//}
//}
