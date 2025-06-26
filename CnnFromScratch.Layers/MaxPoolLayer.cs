using CnnFromScratch.Core;
using System;

namespace CnnFromScratch.Layers
{
    public class MaxPoolLayer : ILayer
    {
        public int PoolSize { get; }
        public int Stride { get; }
        private Tensor3D? _lastInput;
        private (int h, int w)[,,]? _maxIndices; // Remove required, make nullable

        public MaxPoolLayer(int poolSize, int stride)
        {
            if (poolSize <= 0) throw new ArgumentException("Pool size must be positive", nameof(poolSize));
            if (stride <= 0) throw new ArgumentException("Stride must be positive", nameof(stride));

            PoolSize = poolSize;
            Stride = stride;
        }

        public Tensor3D Forward(Tensor3D input)
        {
            _lastInput = input;
            
            int outHeight = (input.Height - PoolSize) / Stride + 1;
            int outWidth = (input.Width - PoolSize) / Stride + 1;
            
            var output = new Tensor3D(input.Channels, outHeight, outWidth);
            _maxIndices = new (int h, int w)[input.Channels, outHeight, outWidth];

            for (int c = 0; c < input.Channels; c++)
            {
                for (int h = 0; h < outHeight; h++)
                {
                    for (int w = 0; w < outWidth; w++)
                    {
                        float maxVal = float.MinValue;
                        int maxH = -1, maxW = -1;

                        // Find maximum in pooling window
                        int startH = h * Stride;
                        int startW = w * Stride;

                        for (int ph = 0; ph < PoolSize && startH + ph < input.Height; ph++)
                        {
                            for (int pw = 0; pw < PoolSize && startW + pw < input.Width; pw++)
                            {
                                int ih = startH + ph;
                                int iw = startW + pw;
                                float val = input[c, ih, iw];
                                
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                    maxH = ih;
                                    maxW = iw;
                                }
                            }
                        }

                        // Ensure we found a valid maximum
                        if (maxH == -1 || maxW == -1)
                        {
                            throw new InvalidOperationException(
                                $"Failed to find maximum value in pooling window at position ({c},{h},{w})");
                        }

                        output[c, h, w] = maxVal;
                        _maxIndices![c, h, w] = (maxH, maxW); // Add null-forgiving operator since we know it's initialized
                    }
                }
            }

            return output;
        }

        public Tensor3D Backward(Tensor3D outputGradient)
        {
            if (_lastInput == null || _maxIndices == null)
                throw new InvalidOperationException("Backward called before Forward");

            var inputGradient = new Tensor3D(_lastInput.Channels, _lastInput.Height, _lastInput.Width);

            // Handle flattened gradient from Dense layer
            if (outputGradient.Channels * outputGradient.Height * outputGradient.Width == 
                _maxIndices.GetLength(0) * _maxIndices.GetLength(1) * _maxIndices.GetLength(2))
            {
                // Reshape the gradient to match the stored indices shape
                var reshapedGradient = new Tensor3D(
                    _maxIndices.GetLength(0),
                    _maxIndices.GetLength(1),
                    _maxIndices.GetLength(2));

                // Copy flattened data to reshaped tensor
                int flatIndex = 0;
                for (int c = 0; c < outputGradient.Channels; c++)
                    for (int h = 0; h < outputGradient.Height; h++)
                        for (int w = 0; w < outputGradient.Width; w++)
                        {
                            int targetC = flatIndex / (_maxIndices.GetLength(1) * _maxIndices.GetLength(2));
                            int remainder = flatIndex % (_maxIndices.GetLength(1) * _maxIndices.GetLength(2));
                            int targetH = remainder / _maxIndices.GetLength(2);
                            int targetW = remainder % _maxIndices.GetLength(2);

                            reshapedGradient[targetC, targetH, targetW] = outputGradient[c, h, w];
                            flatIndex++;
                        }

                outputGradient = reshapedGradient;
            }
            // Original dimension check
            else if (outputGradient.Channels != _maxIndices.GetLength(0) ||
                     outputGradient.Height != _maxIndices.GetLength(1) ||
                     outputGradient.Width != _maxIndices.GetLength(2))
            {
                throw new ArgumentException(
                    $"Output gradient dimensions ({outputGradient.Channels}x{outputGradient.Height}x{outputGradient.Width}) " +
                    $"do not match stored indices dimensions ({_maxIndices.GetLength(0)}x{_maxIndices.GetLength(1)}x{_maxIndices.GetLength(2)})",
                    nameof(outputGradient));
            }

            // Rest of the backward pass remains the same
            try
            {
                for (int c = 0; c < outputGradient.Channels; c++)
                {
                    for (int h = 0; h < outputGradient.Height; h++)
                    {
                        for (int w = 0; w < outputGradient.Width; w++)
                        {
                            var (maxH, maxW) = _maxIndices[c, h, w];
                            
                            if (maxH < 0 || maxH >= _lastInput.Height ||
                                maxW < 0 || maxW >= _lastInput.Width)
                            {
                                throw new InvalidOperationException(
                                    $"Invalid stored index at ({c},{h},{w}): ({maxH},{maxW})");
                            }

                            inputGradient[c, maxH, maxW] += outputGradient[c, h, w];
                        }
                    }
                }
            }
            catch (IndexOutOfRangeException ex)
            {
                throw new InvalidOperationException(
                    $"Index out of range during backward pass. Input: {_lastInput.Channels}x{_lastInput.Height}x{_lastInput.Width}, " +
                    $"Output: {outputGradient.Channels}x{outputGradient.Height}x{outputGradient.Width}", ex);
            }

            return inputGradient;
        }

        public object GetWeights() => Array.Empty<float>();
        public float[] GetBiases() => Array.Empty<float>();
        public void SetWeightsAndBiases(object weights, float[] biases) { }
    }
}