namespace CnnFromScratch.Core
{
    public static class DimensionCalculator
    {
        public static int CalculateConvOutputSize(int inputSize, int kernelSize, int stride, int padding)
        {
            return ((inputSize + 2 * padding - kernelSize) / stride) + 1;
        }

        public static int CalculatePoolOutputSize(int inputSize, int poolSize, int stride)
        {
            return (inputSize - poolSize) / stride + 1;
        }

        public static (int channels, int height, int width) CalculateOutputDimensions(
            int inputChannels, int inputHeight, int inputWidth,
            int kernelSize, int outputChannels, int stride, int padding)
        {
            var outHeight = CalculateConvOutputSize(inputHeight, kernelSize, stride, padding);
            var outWidth = CalculateConvOutputSize(inputWidth, kernelSize, stride, padding);
            return (outputChannels, outHeight, outWidth);
        }

        public static int CalculateFlattenedSize(int channels, int height, int width)
        {
            return channels * height * width;
        }
    }
}