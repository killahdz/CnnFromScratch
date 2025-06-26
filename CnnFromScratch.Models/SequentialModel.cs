using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using CnnFromScratch.Models.Serialization;

namespace CnnFromScratch.Models
{
    [JsonConverter(typeof(SequentialModelConverter))]
    public class SequentialModel
    {
        private readonly List<ILayer> _layersInternal = [];
        public IReadOnlyList<ILayer> Layers => _layersInternal;

        internal IEnumerable<ILayer> GetLayersForSerialization()
        {
            return _layersInternal;
        }

        public void AddLayer(ILayer layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            _layersInternal.Add(layer);
        }

        public Tensor3D Forward(Tensor3D input)
        {
            if (input == null)
                throw new ArgumentNullException(nameof(input));
            if (_layersInternal.Count == 0)
                throw new InvalidOperationException("Model must contain at least one layer.");

            var output = input;
            foreach (var layer in _layersInternal ) { 
                output = layer.Forward(output);
            }
            return output;
        }

        public Tensor4D Forward(Tensor4D batchInput)
        {
            Tensor4D output = batchInput;
            foreach (var layer in Layers)
            {
                output = layer.Forward(output);
            }
            return output;
        }

        public Tensor4D Backward(Tensor4D batchGradient)
        {
            Tensor4D grad = batchGradient;
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                grad = Layers[i].Backward(grad);
            }
            return grad;
        }


    }
}