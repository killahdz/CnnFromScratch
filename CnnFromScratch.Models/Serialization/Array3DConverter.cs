using System.Text.Json;
using System.Text.Json.Serialization;

namespace CnnFromScratch.Models.Serialization
{
    public class Array3DConverter : JsonConverter<float[,,]>
    {
        public override float[,,] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException();

            int dim0 = 0, dim1 = 0, dim2 = 0;
            float[]? data = null;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                    break;

                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException();

                string? propertyName = reader.GetString();
                reader.Read();

                switch (propertyName)
                {
                    case "Dimensions":
                        if (reader.TokenType != JsonTokenType.StartArray)
                            throw new JsonException();
                        
                        reader.Read();
                        dim0 = reader.GetInt32();
                        reader.Read();
                        dim1 = reader.GetInt32();
                        reader.Read();
                        dim2 = reader.GetInt32();
                        reader.Read(); // EndArray
                        break;

                    case "Data":
                        data = MultidimensionalArrayConverterBase.ReadFlattenedArray<float>(ref reader, options);
                        break;
                }
            }

            if (data == null)
                throw new JsonException("Data property not found");

            var result = new float[dim0, dim1, dim2];
            int index = 0;
            for (int i = 0; i < dim0; i++)
                for (int j = 0; j < dim1; j++)
                    for (int k = 0; k < dim2; k++)
                        result[i, j, k] = data[index++];

            return result;
        }

        public override void Write(Utf8JsonWriter writer, float[,,] value, JsonSerializerOptions options)
        {
            MultidimensionalArrayConverterBase.WriteArray<float>(writer, value, options);
        }
    }
}