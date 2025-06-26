using System.Collections.Generic;
using System.Text.Json.Serialization;
using CnnFromScratch.Layers;

namespace CnnFromScratch.Models.Serialization
{
    internal class SequentialModelDto
    {
        [JsonPropertyName("Layers")]
        public List<ILayer> LayersList { get; set; } = new();

        // Factory method to create from model
        public static SequentialModelDto FromModel(SequentialModel model)
        {
            return new SequentialModelDto 
            { 
                LayersList = model.Layers.ToList() 
            };
        }

        // Method to create model from DTO
        public SequentialModel ToModel()
        {
            var model = new SequentialModel();
            foreach (var layer in LayersList)
            {
                model.AddLayer(layer);
            }
            return model;
        }
    }
}