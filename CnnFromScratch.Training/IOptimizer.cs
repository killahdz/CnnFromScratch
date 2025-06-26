using CnnFromScratch.Layers;

public interface IOptimizer
{
    void UpdateLayer(ILayer layer, float learningRate);
}



