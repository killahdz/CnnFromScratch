using CnnFromScratch.Core;

public class Tensor4D
{
    // Dimensions: BatchSize x Channels x Height x Width
    public int BatchSize { get; }
    public int Channels { get; }
    public int Height { get; }
    public int Width { get; }

    // Underlying data storage (flattened)
    private readonly float[] _data;

    public Tensor4D(int batchSize, int channels, int height, int width)
    {
        if (batchSize <= 0 || channels <= 0 || height <= 0 || width <= 0)
            throw new ArgumentException("All dimensions must be positive.");

        BatchSize = batchSize;
        Channels = channels;
        Height = height;
        Width = width;

        _data = new float[BatchSize * Channels * Height * Width];
    }

    // Indexer to access the element at (batch, channel, height, width)
    public float this[int b, int c, int h, int w]
    {
        get
        {
            ValidateIndices(b, c, h, w);
            int index = GetFlatIndex(b, c, h, w);
            return _data[index];
        }
        set
        {
            ValidateIndices(b, c, h, w);
            int index = GetFlatIndex(b, c, h, w);
            _data[index] = value;
        }
    }

    // Extract a single sample as a Tensor3D (the b-th sample)
    public Tensor3D GetSlice(int b)
    {
        if (b < 0 || b >= BatchSize)
            throw new ArgumentOutOfRangeException(nameof(b));

        var slice = new Tensor3D(Channels, Height, Width);

        for (int c = 0; c < Channels; c++)
        {
            for (int h = 0; h < Height; h++)
            {
                for (int w = 0; w < Width; w++)
                {
                    slice[c, h, w] = this[b, c, h, w];
                }
            }
        }

        return slice;
    }

    // Write a Tensor3D back into batch slot b
    public void SetSlice(int b, Tensor3D slice)
    {
        if (b < 0 || b >= BatchSize)
            throw new ArgumentOutOfRangeException(nameof(b));

        if (slice.Channels != Channels || slice.Height != Height || slice.Width != Width)
            throw new ArgumentException("Slice dimensions must match the Tensor4D dimensions.");

        for (int c = 0; c < Channels; c++)
        {
            for (int h = 0; h < Height; h++)
            {
                for (int w = 0; w < Width; w++)
                {
                    this[b, c, h, w] = slice[c, h, w];
                }
            }
        }
    }

    private int GetFlatIndex(int b, int c, int h, int w)
    {
        // Layout: [Batch, Channel, Height, Width]
        return ((b * Channels + c) * Height + h) * Width + w;
    }

    private void ValidateIndices(int b, int c, int h, int w)
    {
        if (b < 0 || b >= BatchSize) throw new IndexOutOfRangeException(nameof(b));
        if (c < 0 || c >= Channels) throw new IndexOutOfRangeException(nameof(c));
        if (h < 0 || h >= Height) throw new IndexOutOfRangeException(nameof(h));
        if (w < 0 || w >= Width) throw new IndexOutOfRangeException(nameof(w));
    }

    // Optional: Fill all elements with a value
    public void Fill(float value)
    {
        for (int i = 0; i < _data.Length; i++)
            _data[i] = value;
    }

    public static Tensor4D Ones(int batch, int channels, int height, int width)
    {
        var t = new Tensor4D(batch, channels, height, width);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        t[b, c, h, w] = 1f;
        return t;
    }

    public static Tensor4D Random(int batch, int channels, int height, int width)
    {
        var rnd = new Random(42);
        var t = new Tensor4D(batch, channels, height, width);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        t[b, c, h, w] = (float)rnd.NextDouble();
        return t;
    }

    public static Tensor4D OnesLike(Tensor4D original)
    {
        return Ones(original.BatchSize, original.Channels, original.Height, original.Width);
    }

    public float[] Flatten()
    {
        var flat = new List<float>();
        for (int b = 0; b < BatchSize; b++)
            for (int c = 0; c < Channels; c++)
                for (int h = 0; h < Height; h++)
                    for (int w = 0; w < Width; w++)
                        flat.Add(this[b, c, h, w]);
        return flat.ToArray();
    }

}
