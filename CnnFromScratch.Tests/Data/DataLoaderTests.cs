using CnnFromScratch.Data;
using NUnit.Framework;
using System.Linq;

namespace CnnFromScratch.Tests.Data
{
    public class DataLoaderTests
    {
        private DataLoader _loader = null!;

        [SetUp]
        public void Setup()
        {
            _loader = new DataLoader();
        }

        [Test]
        public void LoadCifar10_Training_CorrectDimensions()
        {
            // Arrange & Act
            var (images, labels) = _loader.LoadCifar10(isTraining: true);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(images.Length, Is.EqualTo(50000), "Training set should have 50000 images");
                Assert.That(labels.Length, Is.EqualTo(50000), "Labels should match image count");
                Assert.That(images[0].Channels, Is.EqualTo(3), "Images should have 3 channels (RGB)");
                Assert.That(images[0].Height, Is.EqualTo(32), "Images should be 32 pixels high");
                Assert.That(images[0].Width, Is.EqualTo(32), "Images should be 32 pixels wide");
            });
        }

        [Test]
        public void LoadCifar10_Test_CorrectDimensions()
        {
            // Arrange & Act
            var (images, labels) = _loader.LoadCifar10(isTraining: false);

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(images.Length, Is.EqualTo(10000), "Test set should have 10000 images");
                Assert.That(labels.Length, Is.EqualTo(10000), "Labels should match image count");
            });
        }

        [Test]
        public void GetBatches_CorrectBatchSize()
        {
            // Arrange
            var (images, labels) = _loader.LoadCifar10(isTraining: true);
            int batchSize = 128;

            // Act
            var batches = _loader.GetBatches(batchSize).ToList();

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(batches[0].Batch.Length, Is.EqualTo(batchSize));
                Assert.That(batches[0].Labels.Length, Is.EqualTo(batchSize));
                // Last batch might be smaller
                Assert.That(batches.Last().Batch.Length, Is.LessThanOrEqualTo(batchSize));
            });
        }

        [Test]
        public void GetBatches_Shuffle_DifferentOrder()
        {
            // Arrange
            var (images, labels) = _loader.LoadCifar10(isTraining: true);
            int batchSize = 1000;

            // Act
            var firstBatch = _loader.GetBatches(batchSize, shuffle: true).First();
            var secondBatch = _loader.GetBatches(batchSize, shuffle: true).First();

            // Assert
            // Check if at least some elements are in different positions
            bool hasDifference = false;
            for (int i = 0; i < batchSize; i++)
            {
                if (firstBatch.Labels[i] != secondBatch.Labels[i])
                {
                    hasDifference = true;
                    break;
                }
            }
            Assert.That(hasDifference, Is.True, "Shuffled batches should have different order");
        }

        [Test]
        public void LoadCifar10_ImageValues_InValidRange()
        {
            // Arrange & Act
            var (images, _) = _loader.LoadCifar10(isTraining: true);

            // Assert
            var image = images[0];
            for (int c = 0; c < image.Channels; c++)
            {
                for (int h = 0; h < image.Height; h++)
                {
                    for (int w = 0; w < image.Width; w++)
                    {
                        Assert.That(image[c, h, w], Is.InRange(0, 1),
                            "Pixel values should be normalized to [0, 1]");
                    }
                }
            }
        }
    }
}