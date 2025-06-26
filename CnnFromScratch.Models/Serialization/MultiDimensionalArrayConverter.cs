using System.Text.Json;
using System.Text.Json.Serialization;

namespace CnnFromScratch.Models.Serialization
{
    public class MultiDimensionalArrayConverter<T> : JsonConverter<T[,,,]>
    {
        public override T[,,,] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException("Expected StartObject token");

            int dim0 = 0, dim1 = 0, dim2 = 0, dim3 = 0;
            T[]? data = null;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    break;

                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException("Expected PropertyName token");

                string? propertyName = reader.GetString();
                reader.Read();

                switch (propertyName)
                {
                    case "Dimensions":
                        if (reader.TokenType != JsonTokenType.StartArray)
                            throw new JsonException("Expected StartArray token for Dimensions");
                        
                        reader.Read();
                        dim0 = reader.GetInt32();
                        reader.Read();
                        dim1 = reader.GetInt32();
                        reader.Read();
                        dim2 = reader.GetInt32();
                        reader.Read();
                        dim3 = reader.GetInt32();
                        reader.Read(); // EndArray
                        break;

                    case "Data":
                        data = JsonSerializer.Deserialize<T[]>(ref reader, options);
                        break;
                }
            }

            if (data == null)
                throw new JsonException("Data property not found");

            var result = new T[dim0, dim1, dim2, dim3];
            int index = 0;
            for (int i = 0; i < dim0; i++)
                for (int j = 0; j < dim1; j++)
                    for (int k = 0; k < dim2; k++)
                        for (int l = 0; l < dim3; l++)
                            result[i, j, k, l] = data[index++];

            return result;
        }

        public override void Write(Utf8JsonWriter writer, T[,,,] value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            writer.WritePropertyName("Dimensions");
            writer.WriteStartArray();
            writer.WriteNumberValue(value.GetLength(0));
            writer.WriteNumberValue(value.GetLength(1));
            writer.WriteNumberValue(value.GetLength(2));
            writer.WriteNumberValue(value.GetLength(3));
            writer.WriteEndArray();

            writer.WritePropertyName("Data");
            var flattenedData = new T[value.Length];
            int index = 0;
            for (int i = 0; i < value.GetLength(0); i++)
                for (int j = 0; j < value.GetLength(1); j++)
                    for (int k = 0; k < value.GetLength(2); k++)
                        for (int l = 0; l < value.GetLength(3); l++)
                            flattenedData[index++] = value[i, j, k, l];

            JsonSerializer.Serialize(writer, flattenedData, options);

            writer.WriteEndObject();
        }
    }
}