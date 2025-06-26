using System;
using CnnFromScratch.Models;
using CnnFromScratch.Layers;
using CnnFromScratch.Core;

namespace CnnFromScratch.App
{
    public class NetworkVisualizerService
    {
        public void Visualize(SequentialModel model)
        {
            Console.WriteLine("# CNN Structure Visualization");

            int layerIndex = 0;
            foreach (var layer in model.Layers)
            {
                Console.WriteLine($"\n* Layer {layerIndex++}: {layer.GetType().Name}");

                var weights = layer.GetWeights();
                var biases = layer.GetBiases();

                if (weights is float[,,,] convWeights)
                {
                    // Convert float[,,,] to Tensor3D[] for visualization
                    int outputChannels = convWeights.GetLength(0);
                    var kernels = new Tensor3D[outputChannels];
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        int inputChannels = convWeights.GetLength(1);
                        int kernelHeight = convWeights.GetLength(2);
                        int kernelWidth = convWeights.GetLength(3);
                        var kernel = new Tensor3D(inputChannels, kernelHeight, kernelWidth);
                        for (int ic = 0; ic < inputChannels; ic++)
                            for (int h = 0; h < kernelHeight; h++)
                                for (int w = 0; w < kernelWidth; w++)
                                    kernel[ic, h, w] = convWeights[oc, ic, h, w];
                        kernels[oc] = kernel;
                    }

                    Console.WriteLine($"  Filters: {kernels.Length} filters");
                    for (int i = 0; i < kernels.Length; i++)
                    {
                        Console.WriteLine($"    Filter {i}:");
                        PrintWeightSlice(kernels[i]);
                    }
                }
                else if (weights is Matrix matrix)
                {
                    Console.WriteLine($"  Dense Weights: {matrix.Rows}x{matrix.Cols}");
                    matrix.Print($"Weights");
                }
                else if (weights is float[] gamma)
                {
                    Console.WriteLine($"  Gamma (Scale) Parameters: {gamma.Length}");
                    for (int i = 0; i < gamma.Length; i++)
                        Console.Write($"{gamma[i]:F2} ");
                    Console.WriteLine();
                }

                if (biases?.Length > 0)
                {
                    Console.WriteLine("  Biases:");
                    for (int i = 0; i < biases.Length; i++)
                        Console.Write($"{biases[i]:F2} ");
                    Console.WriteLine();
                }
            }

            Console.WriteLine("\n# Rendering complete (text mode).");
        }

        private void PrintWeightSlice(Tensor3D kernel)
        {
            for (int c = 0; c < kernel.Channels; c++)
            {
                Console.WriteLine($"    Channel {c}:");
                for (int h = 0; h < kernel.Height; h++)
                {
                    for (int w = 0; w < kernel.Width; w++)
                    {
                        float val = kernel[c, h, w];
                        Console.ForegroundColor = GetColorForWeight(val);
                        Console.Write($"{val,6:F2} ");
                        Console.ResetColor();
                    }
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }

        private ConsoleColor GetColorForWeight(float value)
        {
            if (value > 0.5f) return ConsoleColor.Blue;
            if (value > 0.1f) return ConsoleColor.Cyan;
            if (value < -0.5f) return ConsoleColor.Red;
            if (value < -0.1f) return ConsoleColor.DarkRed;
            return ConsoleColor.Gray;
        }
    }
}