using CnnFromScratch.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace CnnFromScratch.Data
{
    public class DataLoader
    {
        private const string CifarUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
        private const string DefaultDataDirectory = "cifar10_data";
        private const string ArchiveFileName = "cifar10.tar.gz";
        private const string DataBatchPattern = "data_batch_*";
        private const string TestBatchFile = "test_batch*";
        private const int ImageSize = 32;
        private const int ChannelSize = ImageSize * ImageSize;
        private const int PixelDepth = 255;

        private readonly string _dataDirectory;
        private Tensor3D[]? _images;
        private int[]? _labels;

        public DataLoader(string? dataDirectory = null)
        {
            _dataDirectory = dataDirectory ?? Path.Combine("c:", "training", DefaultDataDirectory);
        }

        public async Task DownloadAndExtractAsync()
        {
            try
            {
                // Always ensure directory exists
                Directory.CreateDirectory(_dataDirectory);

                // If we have the data files, we're done
                if (HasRequiredDataFiles())
                {
                    return;
                }

                var archivePath = Path.Combine(_dataDirectory, ArchiveFileName);

                // Download if needed
                if (!File.Exists(archivePath))
                {
                    using var client = new HttpClient();
                    var response = await client.GetAsync(CifarUrl);
                    response.EnsureSuccessStatusCode();
                    using var fs = new FileStream(archivePath, FileMode.Create);
                    await response.Content.CopyToAsync(fs);
                }

                // After download/existing archive check, verify data files again
                if (!HasRequiredDataFiles())
                {
                    throw new InvalidOperationException(
                        $"CIFAR-10 data files not found in {Path.GetFullPath(_dataDirectory)}. " +
                        $"Please extract {ArchiveFileName} manually to this directory.");
                }
            }
            catch (HttpRequestException ex)
            {
                throw new InvalidOperationException(
                    "Failed to download CIFAR-10 dataset. Please check your internet connection.", ex);
            }
            catch (IOException ex)
            {
                throw new InvalidOperationException(
                    "Failed to access the data directory or files. Please check file permissions.", ex);
            }
        }

        private bool HasRequiredDataFiles()
        {
            try
            {
                // Check for training batches (should be 5 files)
                var trainingFiles = Directory.GetFiles(_dataDirectory, DataBatchPattern);
                if (trainingFiles.Length != 5) return false;

                // Check for test batch
                var testFiles = Directory.GetFiles(_dataDirectory, TestBatchFile);
                if (testFiles.Length != 1) return false;

                // Verify files are readable and non-empty
                foreach (var file in trainingFiles.Concat(testFiles))
                {
                    using var stream = File.OpenRead(file);
                    if (stream.Length == 0) return false;
                }

                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        public virtual (Tensor3D[] Images, int[] Labels) LoadCifar10(bool isTraining = true)
        {
            if (_images != null && _labels != null)
                return (_images, _labels);

            var filePattern = isTraining ? DataBatchPattern : TestBatchFile;
            var files = Directory.GetFiles(_dataDirectory, filePattern);

            var images = new List<Tensor3D>();
            var labels = new List<int>();

            foreach (var file in files)
            {
                using var reader = new BinaryReader(File.OpenRead(file));
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    // Read label (1 byte)
                    labels.Add(reader.ReadByte());

                    // Read image data (3072 bytes = 32x32x3)
                    var imageData = reader.ReadBytes(3 * ChannelSize);
                    var tensor = new Tensor3D(3, ImageSize, ImageSize);

                    // Convert to RGB Tensor3D
                    for (int c = 0; c < 3; c++)
                    {
                        for (int h = 0; h < ImageSize; h++)
                        {
                            for (int w = 0; w < ImageSize; w++)
                            {
                                // Normalize pixel values to [0, 1]
                                tensor[c, h, w] = imageData[c * ChannelSize + h * ImageSize + w] / PixelDepth;
                            }
                        }
                    }
                    images.Add(tensor);
                }
            }

            _images = images.ToArray();
            _labels = labels.ToArray();
            return (_images, _labels);
        }

        public virtual IEnumerable<(Tensor3D[] Batch, int[] Labels)> GetBatches(int batchSize, bool shuffle = true)
        {
            if (_images == null || _labels == null)
                throw new InvalidOperationException("Call LoadCifar10 before getting batches");

            var indices = Enumerable.Range(0, _images.Length).ToArray();
            if (shuffle)
            {
                var random = new Random();
                indices = indices.OrderBy(x => random.Next()).ToArray();
            }

            for (int i = 0; i < _images.Length; i += batchSize)
            {
                var batchIndices = indices.Skip(i).Take(batchSize).ToArray();
                var batchImages = batchIndices.Select(idx => _images[idx]).ToArray();
                var batchLabels = batchIndices.Select(idx => _labels[idx]).ToArray();
                yield return (batchImages, batchLabels);
            }
        }

        public void Shuffle()
        {
            if (_images == null || _labels == null)
                throw new InvalidOperationException("No data loaded to shuffle");

            var random = new Random();
            var n = _images.Length;
            while (n > 1)
            {
                n--;
                var k = random.Next(n + 1);
                (_images[k], _images[n]) = (_images[n], _images[k]);
                (_labels[k], _labels[n]) = (_labels[n], _labels[k]);
            }
        }
    }
}