using CnnFromScratch.Core;
using CnnFromScratch.Data;
using CnnFromScratch.Layers;
using CnnFromScratch.Models;
using CnnFromScratch.Models.Serialization;

namespace CnnFromScratch.Training
{
    public static class TrainerRunner
    {
        public static SequentialModel TrainCifarModel(int epochs = 50, float initialLearningRate = 0.005f, int batchSize = 32, float dropoutRate = 0.3f)
        {
            Console.WriteLine("CIFAR-10 CNN Training");
            Console.WriteLine("=====================");
            var model = CreateCifarModel(dropoutRate);
            var loss = new CrossEntropyLoss();
            var optimizer = new SGDOptimizer(momentum: 0.9f, clipValue: 0.9f);
            var dataLoader = new DataLoader();
            Console.WriteLine("Loading CIFAR-10 dataset...");
            dataLoader.DownloadAndExtractAsync().Wait();
            var (trainImages, trainLabels) = dataLoader.LoadCifar10(true);
            var trainer = new Trainer(model, optimizer, loss, batchSize);
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float lr = initialLearningRate * (float)Math.Pow(0.98, epoch);
                Console.WriteLine($"\nEpoch {epoch + 1}/{epochs} (lr={lr:E3})");
                trainer.Train(dataLoader, 1, lr);
                float acc = trainer.Evaluate(dataLoader);
                Console.WriteLine($"Validation Accuracy: {acc:P2}");
            }
            ModelSerializer.Save(model, "cifar10_model.json");
            return model;
        }

        public static SequentialModel CreateCifarModel(float dropoutRate = 0.3f)
        {
            var model = new SequentialModel();
            // Block 1
            model.AddLayer(new Conv2DLayer(3, 32, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new Conv2DLayer(32, 32, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new MaxPoolLayer(2, 2));
            // Block 2
            model.AddLayer(new Conv2DLayer(32, 64, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new Conv2DLayer(64, 64, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new MaxPoolLayer(2, 2));
            // Block 3
            model.AddLayer(new Conv2DLayer(64, 128, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new Conv2DLayer(128, 128, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new MaxPoolLayer(2, 2));
            // Fully Connected Layers
           // model.AddLayer(new DropoutLayer(dropoutRate));
            model.AddLayer(new DenseLayer(4 * 4 * 128, 512));
            model.AddLayer(new ReLULayer());
          //  model.AddLayer(new DropoutLayer(dropoutRate));
            model.AddLayer(new DenseLayer(512, 10));
            model.AddLayer(new SoftmaxLayer());
            return model;
        }

        public static SequentialModel CreateVGG11Model()
        {
            var model = new SequentialModel();
            // Block 1
            model.AddLayer(new Conv2DLayer(3, 64, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new MaxPoolLayer(2, 2));
            // Block 2
            model.AddLayer(new Conv2DLayer(64, 128, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new MaxPoolLayer(2, 2));
            // Block 3
            model.AddLayer(new Conv2DLayer(128, 256, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new Conv2DLayer(256, 256, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new MaxPoolLayer(2, 2));
            // Block 4
            model.AddLayer(new Conv2DLayer(256, 512, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new Conv2DLayer(512, 512, 3, 1, 1));
            model.AddLayer(new BatchNormLayer());
            model.AddLayer(new ReLULayer());
            model.AddLayer(new MaxPoolLayer(2, 2));
            // Fully Connected
         //   model.AddLayer(new DropoutLayer(0.3f));
            model.AddLayer(new DenseLayer(2 * 2 * 512, 512));
            model.AddLayer(new ReLULayer());
         //   model.AddLayer(new DropoutLayer(0.3f));
            model.AddLayer(new DenseLayer(512, 10));
            model.AddLayer(new SoftmaxLayer());
            return model;
        }
    }
}