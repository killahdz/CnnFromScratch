using CnnFromScratch.Core;
using CnnFromScratch.Data;
using CnnFromScratch.Layers;
using CnnFromScratch.Models;
using CnnFromScratch.Models.Serialization;
using CnnFromScratch.Training;
using System;
using System.Threading.Tasks;

namespace CnnFromScratch.App
{
    class Program
    {
        static async Task Main(string[] args)
        {
            try
            {
                Console.WriteLine("CIFAR-10 CNN Training");
                Console.WriteLine("=====================");
                Console.WriteLine("Choose model architecture:");
                Console.WriteLine("(s) Simple 3x(C-B-R-C-B-R-M)=>D=>R=>D=>S");
                Console.WriteLine("(v) VGG11");

                char modelKey;
                do
                {
                    modelKey = char.ToLower(Console.ReadKey().KeyChar);
                    Console.WriteLine();
                } while (modelKey != 's' && modelKey != 'v');

                var model = modelKey == 'v'
                    ? TrainerRunner.CreateVGG11Model()
                    : TrainerRunner.CreateCifarModel(dropoutRate: 0.3f); // Default until optKey is selected

                Console.WriteLine("Choose optimizer:");
                Console.WriteLine("(s) SGD (with momentum)");
                Console.WriteLine("(a) Adam");

                char optKey;
                do
                {
                    optKey = char.ToLower(Console.ReadKey().KeyChar);
                    Console.WriteLine();
                } while (optKey != 's' && optKey != 'a');

                var (initialLearningRate, batchSize, dropoutRate, epochs) = PromptForHyperparameters(optKey);

                // Recreate model with selected dropoutRate if using CifarModel
                if (modelKey == 's')
                {
                    model = TrainerRunner.CreateCifarModel(dropoutRate: dropoutRate);
                }

                IOptimizer optimizer = optKey == 'a'
                    ? new AdamOptimizer()
                    : new SGDOptimizer(momentum: 0.9f, clipValue: 0.9f);

                var visualizer = new NetworkVisualizerService();
                // visualizer.Visualize(model); // optional

                Console.WriteLine("Loading CIFAR-10 dataset...");
                var dataLoader = new DataLoader();
                await dataLoader.DownloadAndExtractAsync();
                var (trainImages, trainLabels) = dataLoader.LoadCifar10(isTraining: true);

                if (trainImages == null || trainImages.Length == 0)
                {
                    throw new InvalidOperationException("No training images were loaded. Check CIFAR-10 data.");
                }

                Console.WriteLine($"Loaded {trainImages.Length} training images");
                VerifyDataRange(trainImages[0]);

                var trainer = new Trainer(model, optimizer, new CrossEntropyLoss(), batchSize);

                Console.WriteLine("\nStarting training...");
                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    float lr = initialLearningRate * (float)Math.Pow(0.98, epoch);
                    Console.WriteLine($"\nEpoch {epoch + 1}/{epochs} (lr={lr:E3})");
                    trainer.Train(dataLoader, epochs: 1, lr);
                    float accuracy = trainer.Evaluate(dataLoader);
                    Console.WriteLine($"Validation Accuracy: {accuracy:P2}");
                }

                ModelSerializer.Save(model, "cifar10_model.json");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine(ex.StackTrace);
            }
        }

        private static (float learningRate, int batchSize, float dropoutRate, int epochs) PromptForHyperparameters(char optimizerKey)
        {
            Console.WriteLine("Select hyperparameter preset (LR is initial, decayed by 0.98 per epoch):");
            if (optimizerKey == 'a')
            {
                Console.WriteLine("(1) Adam - Phase 3 (LR=0.001, Batch=32, Dropout=0.3, Epochs=50, Augmentation)");
            }
            else
            {
                Console.WriteLine("(1) SGD - Phase 1, LR=0.002 (LR=0.002, Batch=32, Dropout=0.3, Epochs=10)");
                Console.WriteLine("(2) SGD - Phase 1, LR=0.005 (LR=0.005, Batch=32, Dropout=0.3, Epochs=10)");
                Console.WriteLine("(3) SGD - Phase 1, LR=0.01 (LR=0.01, Batch=32, Dropout=0.3, Epochs=10)");
                Console.WriteLine("(4) SGD - Phase 2, Dropout=0.2 (LR=0.005, Batch=32, Dropout=0.2, Epochs=20)");
                Console.WriteLine("(5) SGD - Phase 2, Dropout=0.4 (LR=0.005, Batch=32, Dropout=0.4, Epochs=20)");
                Console.WriteLine("(6) SGD - Phase 2, Batch=64 (LR=0.005, Batch=64, Dropout=0.3, Epochs=20)");
            }

            char key;
            do
            {
                key = Console.ReadKey().KeyChar;
                Console.WriteLine();
            } while (!(key >= '1' && key <= (optimizerKey == 'a' ? '1' : '6')));

            return optimizerKey == 'a' ? key switch
            {
                '1' => (0.001f, 32, 0.3f, 50),
                _ => (0.001f, 32, 0.3f, 50)
            } : key switch
            {
                '1' => (0.002f, 32, 0.3f, 10),
                '2' => (0.005f, 32, 0.3f, 10),
                '3' => (0.01f, 32, 0.3f, 10),
                '4' => (0.005f, 32, 0.2f, 20),
                '5' => (0.005f, 32, 0.4f, 20),
                '6' => (0.005f, 64, 0.3f, 20),
                _ => (0.005f, 32, 0.3f, 10)
            };
        }

        private static void VerifyDataRange(Tensor3D image)
        {
            float min = float.MaxValue;
            float max = float.MinValue;

            for (int c = 0; c < image.Channels; c++)
                for (int h = 0; h < image.Height; h++)
                    for (int w = 0; w < image.Width; w++)
                    {
                        float val = image[c, h, w];
                        min = Math.Min(min, val);
                        max = Math.Max(max, val);
                    }

            Console.WriteLine($"Data range: [{min:F3}, {max:F3}]");
        }
    }
}