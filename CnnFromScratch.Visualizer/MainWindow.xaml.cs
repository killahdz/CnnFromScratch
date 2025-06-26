using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using CnnFromScratch.Models;
using CnnFromScratch.Core;
using CnnFromScratch.Layers;
using Matrix = CnnFromScratch.Core.Matrix;
using CnnFromScratch.Training;

namespace CnnFromScratch.Visualizer
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            // Load a model
            var model = TrainerRunner.CreateCifarModel();
            
            DrawModel(model);
        }

        private void DrawModel(SequentialModel model)
        {
            double zOffset = 0;
            const double spacing = 20;

            foreach (var layer in model.Layers)
            {
                if (layer.GetWeights() is float[,,,] convWeights)
                {
                    int filters = convWeights.GetLength(0);
                    for (int i = 0; i < filters; i++)
                    {
                        var cube = CreateColoredCube(zOffset + i * 1.5, 0, 0, GetColorForWeight(convWeights[i, 0, 0, 0]));
                        Viewport.Children.Add(cube);
                    }
                    zOffset += spacing;
                }
                else if (layer.GetWeights() is Matrix dense)
                {
                    var cube = CreateColoredCube(0, 0, zOffset, GetColorForWeight(dense[0, 0]));
                    Viewport.Children.Add(cube);
                    zOffset += spacing;
                }
            }
        }

        private ModelVisual3D CreateColoredCube(double x, double y, double z, Color color)
        {
            var mesh = new MeshGeometry3D();

            Point3D[] corners = new[]
            {
                new Point3D(x-1, y-1, z-1), new Point3D(x+1, y-1, z-1),
                new Point3D(x+1, y+1, z-1), new Point3D(x-1, y+1, z-1),
                new Point3D(x-1, y-1, z+1), new Point3D(x+1, y-1, z+1),
                new Point3D(x+1, y+1, z+1), new Point3D(x-1, y+1, z+1)
            };

            int[] triangleIndices = {
                0,1,2, 0,2,3, 4,6,5, 4,7,6,
                0,4,5, 0,5,1, 1,5,6, 1,6,2,
                2,6,7, 2,7,3, 3,7,4, 3,4,0
            };

            foreach (var idx in triangleIndices)
                mesh.Positions.Add(corners[idx]);

            mesh.TriangleIndices = new Int32Collection(triangleIndices);

            var material = new DiffuseMaterial(new SolidColorBrush(color));
            var model = new GeometryModel3D(mesh, material);
            return new ModelVisual3D { Content = model };
        }

        private Color GetColorForWeight(float weight)
        {
            if (weight > 0.5f) return Colors.Blue;
            if (weight > 0.1f) return Colors.LightBlue;
            if (weight < -0.5f) return Colors.Red;
            if (weight < -0.1f) return Colors.OrangeRed;
            return Colors.Gray;
        }

        private SequentialModel LoadYourSequentialModel()
        {
            // TODO: Load from your trainer/model instance
            return new SequentialModel(); // placeholder
        }
    }
}
