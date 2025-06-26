using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CnnFromScratch.Models.Serialization
{
    public static class ModelSerializer
    {
        private static readonly JsonSerializerOptions _options = new()
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Converters = 
            {
                new LayerConverter(),
                new Array3DConverter(),
                new MultiDimensionalArrayConverter<float>(),
                new Matrix2DConverter(),
                new JsonStringEnumConverter()
            }
        };

        public static void Save(SequentialModel model, string path)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            if (string.IsNullOrEmpty(path))
                throw new ArgumentException("Path cannot be null or empty", nameof(path));

            try
            {
                var dto = SequentialModelDto.FromModel(model);
                var json = JsonSerializer.Serialize(dto, typeof(SequentialModelDto), _options);
                File.WriteAllText(path, json);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to serialize model: {ex.Message}", ex);
            }
        }

        public static SequentialModel Load(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Model file not found", path);

            try
            {
                var json = File.ReadAllText(path);
                var dto = JsonSerializer.Deserialize<SequentialModelDto>(json, _options);

                if (dto == null)
                    throw new InvalidOperationException("Failed to deserialize model");

                return dto.ToModel();
            }
            catch (JsonException ex)
            {
                throw new InvalidOperationException($"Failed to deserialize model: {ex.Message}", ex);
            }
        }
    }
}