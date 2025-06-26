using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace CnnFromScratch.Models.Serialization
{
    public class Matrix2DConverter : JsonConverter<float[,]>
    {
        public override float[,] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException("Expected start of object for matrix deserialization");

            int rows = 0, cols = 0;
            float[]? data = null;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    break;

                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException("Expected property name in matrix");

                string? propertyName = reader.GetString();
                reader.Read();

                switch (propertyName)
                {
                    case "Rows":
                        rows = reader.GetInt32();
                        break;
                    case "Columns":
                        cols = reader.GetInt32();
                        break;
                    case "Data":
                        if (reader.TokenType != JsonTokenType.StartArray)
                            throw new JsonException("Expected start of array for matrix data");

                        var tempList = new List<float>();
                        while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                        {
                            if (reader.TokenType == JsonTokenType.Number)
                            {
                                tempList.Add(reader.GetSingle());
                            }
                        }
                        data = tempList.ToArray();
                        break;
                }
            }

            if (data == null || rows == 0 || cols == 0)
                throw new JsonException("Invalid matrix data: missing dimensions or data");

            if (data.Length != rows * cols)
                throw new JsonException($"Matrix data length mismatch: expected {rows * cols}, got {data.Length}");

            var matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = data[i * cols + j];
                }
            }

            return matrix;
        }

        public override void Write(Utf8JsonWriter writer, float[,] value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            // Write dimensions
            int rows = value.GetLength(0);
            int cols = value.GetLength(1);
            writer.WriteNumber("Rows", rows);
            writer.WriteNumber("Columns", cols);

            // Write flattened data array
            writer.WriteStartArray("Data");
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    writer.WriteNumberValue(value[i, j]);
                }
            }
            writer.WriteEndArray();

            writer.WriteEndObject();
        }
    }
}