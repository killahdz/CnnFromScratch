using CnnFromScratch.Core;
using System;

namespace CnnFromScratch.Training
{
    /// <summary>
    /// Cross-entropy loss implementation for multi-class classification.
    /// Commonly used with softmax activation.
    /// 
    /// Loss = -sum(y_true * log(y_pred))
    /// where y_true is one-hot encoded target and y_pred are predicted probabilities
    /// </summary>
    public class CrossEntropyLoss : ILoss
    {
        private const float Epsilon = 1e-5f;
        private readonly bool _withSoftmax;

        public CrossEntropyLoss(bool withSoftmax = true)
        {
            _withSoftmax = withSoftmax;
        }

        public float Calculate(Tensor3D predicted, Tensor3D actual)
        {
            ValidateInputs(predicted, actual);
            float loss = 0;
            
            for (int c = 0; c < predicted.Channels; c++)
            {
                for (int h = 0; h < predicted.Height; h++)
                {
                    for (int w = 0; w < predicted.Width; w++)
                    {
                        float p = Math.Max(predicted[c, h, w], Epsilon);
                        p = Math.Min(p, 1.0f - Epsilon);
                        float y = actual[c, h, w];
                        
                        if (y > 0)
                        {
                            loss -= y * (float)Math.Log(p);
                        }
                    }
                }
            }

            return loss;
        }

        public Tensor3D Gradient(Tensor3D predicted, Tensor3D actual)
        {
            ValidateInputs(predicted, actual);
            var gradient = new Tensor3D(predicted.Channels, predicted.Height, predicted.Width);

            for (int c = 0; c < predicted.Channels; c++)
            {
                for (int h = 0; h < predicted.Height; h++)
                {
                    for (int w = 0; w < predicted.Width; w++)
                    {
                        float p = predicted[c, h, w];
                        float y = actual[c, h, w];

                        if (_withSoftmax)
                        {
                            // When used with softmax, gradient simplifies to p - y
                            gradient[c, h, w] = p - y;
                        }
                        else
                        {
                            // Pure cross-entropy gradient is -y/p
                            if (y > 0 && p < Epsilon)
                            {
                                gradient[c, h, w] = -1.0f / Epsilon; // Large negative gradient
                            }
                            else if (y == 0 && p > 1 - Epsilon)
                            {
                                gradient[c, h, w] = 1.0f / Epsilon; // Large positive gradient
                            }
                            else
                            {
                                gradient[c, h, w] = y == 0 ? 0 : -y / Math.Max(p, Epsilon);
                            }
                        }
                    }
                }
            }

            return gradient;
        }

        private void ValidateInputs(Tensor3D predicted, Tensor3D actual)
        {
            if (predicted == null || actual == null)
                throw new ArgumentNullException(predicted == null ? nameof(predicted) : nameof(actual));

            if (predicted.Channels != actual.Channels ||
                predicted.Height != actual.Height ||
                predicted.Width != actual.Width)
            {
                throw new ArgumentException(
                    $"Dimension mismatch: predicted {predicted.Channels}x{predicted.Height}x{predicted.Width} " +
                    $"vs actual {actual.Channels}x{actual.Height}x{actual.Width}");
            }

            // Validate probabilities
            for (int c = 0; c < predicted.Channels; c++)
            {
                for (int h = 0; h < predicted.Height; h++)
                {
                    float predSum = 0;
                    float actualSum = 0;
                    
                    for (int w = 0; w < predicted.Width; w++)
                    {
                        float pValue = predicted[c, h, w];
                        float aValue = actual[c, h, w];
                        
                        // Check probability bounds
                        if (pValue < 0 || pValue > 1)
                            throw new ArgumentException("Predicted values must be probabilities between 0 and 1");
                        if (aValue < 0 || aValue > 1)
                            throw new ArgumentException("One-hot encoded values must be between 0 and 1");
                        
                        predSum += pValue;
                        actualSum += aValue;
                    }
                
                    // Allow small numerical error in sums
                    if (Math.Abs(predSum - 1) > Epsilon)
                        throw new ArgumentException("Predicted probabilities must sum to 1");
                    if (Math.Abs(actualSum - 1) > Epsilon)
                        throw new ArgumentException("One-hot encoded values must sum to 1");
                }
            }
        }
    }
}