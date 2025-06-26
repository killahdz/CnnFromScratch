// Loss functions
using CnnFromScratch.Core;

public interface ILoss
{
    float Calculate(Tensor3D predicted, Tensor3D actual);
    Tensor3D Gradient(Tensor3D predicted, Tensor3D actual);
}

