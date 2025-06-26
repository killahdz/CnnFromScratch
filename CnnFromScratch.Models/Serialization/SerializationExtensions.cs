using System.Text.Json;
using System.Text.Json.Serialization;

namespace CnnFromScratch.Models.Serialization
{
    internal static class SerializationExtensions
    {
        internal static JsonSerializerOptions CreateModelSerializerOptions()
        {
            return new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters =
                {
                    new LayerConverter(),
                    new MultiDimensionalArrayConverter<float>(),
                    new JsonStringEnumConverter()
                }
            };
        }
    }
}