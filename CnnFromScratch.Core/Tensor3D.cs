using System;
using System.Text.Json.Serialization;

namespace CnnFromScratch.Core
{
    public class Tensor3D
    {
        [JsonInclude]
        public int Channels { get; private set; }
        
        [JsonInclude]
        public int Height { get; private set; }
        
        [JsonInclude]
        public int Width { get; private set; }

        [JsonInclude]
        private float[,,] Data { get; set; }

        public Tensor3D(int channels, int height, int width)
        {
            if (channels <= 0 || height <= 0 || width <= 0)
                throw new ArgumentException("Tensor dimensions must be greater than 0.");

            Channels = channels;
            Height = height;
            Width = width;
            Data = new float[channels, height, width];
        }

        public float this[int c, int h, int w]
        {
            get => Data[c, h, w];
            set => Data[c, h, w] = value;
        }

        public void Fill(float value)
        {
            for (int c = 0; c < Channels; c++)
                for (int h = 0; h < Height; h++)
                    for (int w = 0; w < Width; w++)
                        Data[c, h, w] = value;
        }

        public void Randomize(float min = -1f, float max = 1f)
        {
            var rand = new Random();
            for (int c = 0; c < Channels; c++)
                for (int h = 0; h < Height; h++)
                    for (int w = 0; w < Width; w++)
                        Data[c, h, w] = (float)(rand.NextDouble() * (max - min) + min);
        }

        public Tensor3D Clone()
        {
            var clone = new Tensor3D(Channels, Height, Width);
            for (int c = 0; c < Channels; c++)
                for (int h = 0; h < Height; h++)
                    for (int w = 0; w < Width; w++)
                        clone[c, h, w] = this[c, h, w];
            return clone;
        }

        public void Print(string label = "Tensor3D")
        {
            Console.WriteLine($"\n{label} (Channels={Channels}, Height={Height}, Width={Width}):");
            for (int c = 0; c < Channels; c++)
            {
                Console.WriteLine($"Channel {c}:");
                for (int h = 0; h < Height; h++)
                {
                    for (int w = 0; w < Width; w++)
                        Console.Write($"{Data[c, h, w]:F2}\t");
                    Console.WriteLine();
                }
                Console.WriteLine();
            }
        }
    }
}
