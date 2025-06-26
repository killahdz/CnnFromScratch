using System.Text.Json;

namespace CnnFromScratch.Models.Serialization
{
    public abstract class MultidimensionalArrayConverterBase
    {
        public static void WriteArray<T>(Utf8JsonWriter writer, Array array, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            
            // Write dimensions
            writer.WritePropertyName("Dimensions");
            writer.WriteStartArray();
            for (int i = 0; i < array.Rank; i++)
            {
                writer.WriteNumberValue(array.GetLength(i));
            }
            writer.WriteEndArray();

            // Write flattened data
            writer.WritePropertyName("Data");
            writer.WriteStartArray();
            foreach (T item in array)
            {
                JsonSerializer.Serialize(writer, item, options);
            }
            writer.WriteEndArray();

            writer.WriteEndObject();
        }

        public static T[] ReadFlattenedArray<T>(ref Utf8JsonReader reader, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartArray)
                throw new JsonException("Expected start of array");

            var list = new List<T>();
            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndArray)
                    break;

                list.Add(JsonSerializer.Deserialize<T>(ref reader, options)!);
            }

            return list.ToArray();
        }
    }
}