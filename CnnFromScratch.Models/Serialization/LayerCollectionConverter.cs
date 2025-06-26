using System.Text.Json;
using System.Text.Json.Serialization;
using CnnFromScratch.Layers;

namespace CnnFromScratch.Models.Serialization
{
    public class LayerCollectionConverter : JsonConverter<List<ILayer>>
    {
        public override List<ILayer>? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartArray)
            {
                throw new JsonException($"Expected array start, got {reader.TokenType}");
            }

            var layers = new List<ILayer>();
            var layerConverter = (JsonConverter<ILayer>)options.GetConverter(typeof(ILayer));

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndArray)
                {
                    break;
                }

                // Each layer should be an object
                if (reader.TokenType != JsonTokenType.StartObject)
                {
                    throw new JsonException($"Expected object start, got {reader.TokenType}");
                }

                var layer = layerConverter.Read(ref reader, typeof(ILayer), options);
                if (layer != null)
                {
                    layers.Add(layer);
                }
            }

            return layers;
        }

        public override void Write(Utf8JsonWriter writer, List<ILayer> value, JsonSerializerOptions options)
        {
            if (value == null)
            {
                writer.WriteNullValue();
                return;
            }

            writer.WriteStartArray();

            var layerConverter = (JsonConverter<ILayer>)options.GetConverter(typeof(ILayer));
            foreach (var layer in value)
            {
                if (layer != null)
                {
                    layerConverter.Write(writer, layer, options);
                }
            }

            writer.WriteEndArray();
        }

        public override bool CanConvert(Type typeToConvert)
        {
            return typeToConvert == typeof(List<ILayer>) || 
                   typeToConvert == typeof(IReadOnlyList<ILayer>);
        }
    }
}