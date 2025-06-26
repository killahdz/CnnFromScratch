using System;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Collections.Generic;
using CnnFromScratch.Layers;

namespace CnnFromScratch.Models.Serialization
{
    public class SequentialModelConverter : JsonConverter<SequentialModel>
    {
        public override bool CanConvert(Type typeToConvert)
        {
            return typeToConvert == typeof(SequentialModel);
        }

        public override SequentialModel Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException("Expected start of object");

            var model = new SequentialModel();
            
            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    break;

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    string? propertyName = reader.GetString();
                    reader.Read();

                    if (propertyName == "Layers")
                    {
                        if (reader.TokenType != JsonTokenType.StartArray)
                            throw new JsonException("Expected start of array for Layers");

                        var layerConverter = options.GetConverter(typeof(ILayer)) as JsonConverter<ILayer>;
                        if (layerConverter == null)
                            throw new JsonException("No converter found for ILayer");

                        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                        {
                            var layer = layerConverter.Read(ref reader, typeof(ILayer), options);
                            if (layer != null)
                                model.AddLayer(layer);
                        }
                    }
                }
            }

            return model;
        }

        public override void Write(Utf8JsonWriter writer, SequentialModel value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            writer.WritePropertyName("Layers");
            writer.WriteStartArray();

            var layerConverter = options.GetConverter(typeof(ILayer)) as JsonConverter<ILayer>;
            if (layerConverter == null)
                throw new JsonException("No converter found for ILayer");

            foreach (var layer in value.GetLayersForSerialization())
            {
                if (layer is ILayer typedLayer)
                {
                    layerConverter.Write(writer, typedLayer, options);
                }
                else
                {
                    throw new JsonException("Layer is not of type ILayer");
                }
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }
    }
}