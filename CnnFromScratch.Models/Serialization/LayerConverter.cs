using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using System;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CnnFromScratch.Models.Serialization
{
    public class LayerConverter : JsonConverter<ILayer>
    {
        public override bool CanConvert(Type typeToConvert)
            => typeof(ILayer).IsAssignableFrom(typeToConvert);

        public override ILayer? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException("Expected start of object");

            using var jsonDoc = JsonDocument.ParseValue(ref reader);
            var root = jsonDoc.RootElement;

            var typeName = root.GetProperty("Type").GetString() 
                ?? throw new JsonException("Type property missing or invalid");
            
            var type = Type.GetType(typeName) 
                ?? throw new JsonException($"Cannot find type: {typeName}");

            // Create new options without this converter to prevent recursion
            var localOptions = new JsonSerializerOptions(options);
            for (int i = localOptions.Converters.Count - 1; i >= 0; i--)
            {
                if (localOptions.Converters[i] is LayerConverter)
                {
                    localOptions.Converters.RemoveAt(i);
                }
            }

            // Deserialize the base object without this converter
            var layer = (ILayer)JsonSerializer.Deserialize(root.GetRawText(), type, localOptions)!;

            // Handle weights and biases if present
            if (root.TryGetProperty("Weights", out var weightsElement) &&
                root.TryGetProperty("Biases", out var biasesElement))
            {
                object? weights = null;
                if (type == typeof(Conv2DLayer))
                {
                    if (weightsElement.ValueKind != JsonValueKind.Object)
                        throw new JsonException("Expected array for Conv2DLayer weights");

                    // Parse dimensions
                    var dimensions = weightsElement.GetProperty("Dimensions").EnumerateArray()
                        .Select(x => x.GetInt32())
                        .ToArray();

                    if (dimensions.Length != 4)
                        throw new JsonException("Expected 4 dimensions for Conv2DLayer weights");

                    // Parse data
                    var data = weightsElement.GetProperty("Data").EnumerateArray()
                        .Select(x => x.GetSingle())
                        .ToArray();

                    var arr = new float[dimensions[0], dimensions[1], dimensions[2], dimensions[3]];
                    int index = 0;
                    for (int i = 0; i < dimensions[0]; i++)
                        for (int j = 0; j < dimensions[1]; j++)
                            for (int k = 0; k < dimensions[2]; k++)
                                for (int l = 0; l < dimensions[3]; l++)
                                    arr[i, j, k, l] = data[index++];

                    weights = arr;
                }
                else if (type == typeof(DenseLayer))
                {
                    // Handle Matrix weights for DenseLayer
                    if (weightsElement.ValueKind != JsonValueKind.Object)
                        throw new JsonException("Expected object for DenseLayer weights");

                    var rows = weightsElement.GetProperty("Rows").GetInt32();
                    var cols = weightsElement.GetProperty("Cols").GetInt32();
                    var data = weightsElement.GetProperty("Data")
                        .EnumerateArray()
                        .Select(x => x.GetSingle())
                        .ToArray();

                    var matrix = new Matrix(rows, cols);
                    for (int i = 0; i < rows; i++)
                        for (int j = 0; j < cols; j++)
                            matrix[i, j] = data[i * cols + j];

                    weights = matrix;
                }
                else
                {
                    weights = JsonSerializer.Deserialize(weightsElement.GetRawText(), 
                        type.GetProperty("Weights")?.PropertyType ?? typeof(object), 
                        localOptions);
                }

                var biases = JsonSerializer.Deserialize<float[]>(biasesElement.GetRawText(), localOptions);

                if (weights != null && biases != null)
                {
                    layer.SetWeightsAndBiases(weights, biases);
                }
            }

            return layer;
        }

        public override void Write(Utf8JsonWriter writer, ILayer value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            writer.WriteString("Type", value.GetType().AssemblyQualifiedName);

            foreach (var prop in value.GetType().GetProperties())
            {
                if (!ShouldSerializeProperty(prop)) continue;

                writer.WritePropertyName(prop.Name);
                JsonSerializer.Serialize(writer, prop.GetValue(value), options);
            }

            // Write weights and biases
            var weights = value.GetWeights();
            var biases = value.GetBiases();

            if (weights != null)
            {
                writer.WritePropertyName("Weights");
                if (weights is float[,,,] weightsArray)
                {
                    // Write in MultiDimensionalArrayConverter format
                    writer.WriteStartObject();
                    
                    writer.WritePropertyName("Dimensions");
                    writer.WriteStartArray();
                    writer.WriteNumberValue(weightsArray.GetLength(0));
                    writer.WriteNumberValue(weightsArray.GetLength(1));
                    writer.WriteNumberValue(weightsArray.GetLength(2));
                    writer.WriteNumberValue(weightsArray.GetLength(3));
                    writer.WriteEndArray();

                    writer.WritePropertyName("Data");
                    var flattenedData = new float[weightsArray.Length];
                    int index = 0;
                    for (int i = 0; i < weightsArray.GetLength(0); i++)
                        for (int j = 0; j < weightsArray.GetLength(1); j++)
                            for (int k = 0; k < weightsArray.GetLength(2); k++)
                                for (int l = 0; l < weightsArray.GetLength(3); l++)
                                    flattenedData[index++] = weightsArray[i, j, k, l];

                    JsonSerializer.Serialize(writer, flattenedData, options);
                    
                    writer.WriteEndObject();
                }
                else if (weights is Matrix matrix)
                {
                    writer.WriteStartObject();
                    writer.WriteNumber("Rows", matrix.Rows);
                    writer.WriteNumber("Cols", matrix.Cols);
                    
                    writer.WritePropertyName("Data");
                    writer.WriteStartArray();
                    for (int i = 0; i < matrix.Rows; i++)
                        for (int j = 0; j < matrix.Cols; j++)
                            writer.WriteNumberValue(matrix[i, j]);
                    writer.WriteEndArray();
                    
                    writer.WriteEndObject();
                }
                else
                {
                    JsonSerializer.Serialize(writer, weights, options);
                }
            }

            if (biases != null)
            {
                writer.WritePropertyName("Biases");
                JsonSerializer.Serialize(writer, biases, options);
            }

            writer.WriteEndObject();
        }

        private static bool ShouldSerializeProperty(PropertyInfo prop)
        {
            if (!prop.CanRead) return false;
            if (prop.GetCustomAttribute<JsonIgnoreAttribute>() != null) return false;

            var skipProperties = new[] 
            { 
                "WeightGradients", 
                "BiasGradients", 
                "LastInput", 
                "LastOutput" 
            };

            return !skipProperties.Contains(prop.Name);
        }
    }
}