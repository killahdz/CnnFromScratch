using CnnFromScratch.Core;
using CnnFromScratch.Tests;

public class TestLoss : ILoss
{
    private readonly int _numClasses;
    public int CalculateCallCount { get; private set; }
    public int GradientCallCount { get; private set; }

    public TestLoss(int numClasses = 10) // CIFAR-10 has 10 classes
    {
        if (numClasses <= 0)
            throw new ArgumentException("Number of classes must be positive", nameof(numClasses));
            
        _numClasses = numClasses;
    }

    public float Calculate(Tensor3D predicted, Tensor3D actual)
    {
        ValidateDimensions(predicted, actual);
        CalculateCallCount++;
        return 0.1f; // Return small loss value for testing
    }

    public Tensor3D Gradient(Tensor3D predicted, Tensor3D actual)
    {
        ValidateDimensions(predicted, actual);
        GradientCallCount++;
        
        // Return gradient matching the predicted tensor dimensions
        var gradient = new Tensor3D(predicted.Channels, predicted.Height, predicted.Width);
        gradient.Fill(0.1f); // Small gradient values for stability
        return gradient;
    }

    private void ValidateDimensions(Tensor3D predicted, Tensor3D actual)
    {
        if (predicted.Width != _numClasses)
            throw new ArgumentException(
                $"Predicted tensor width ({predicted.Width}) does not match number of classes ({_numClasses})");

        if (actual.Width != _numClasses)
            throw new ArgumentException(
                $"Actual tensor width ({actual.Width}) does not match number of classes ({_numClasses})");

        if (predicted.Channels != 1 || predicted.Height != 1)
            throw new ArgumentException(
                $"Predicted tensor should be 1x1x{_numClasses}, got {predicted.Channels}x{predicted.Height}x{predicted.Width}");

        if (actual.Channels != 1 || actual.Height != 1)
            throw new ArgumentException(
                $"Actual tensor should be 1x1x{_numClasses}, got {actual.Channels}x{actual.Height}x{actual.Width}");
    }
}