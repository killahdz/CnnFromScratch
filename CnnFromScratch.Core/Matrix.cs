using System;

namespace CnnFromScratch.Core
{
    public class Matrix
    {
        public int Rows { get; }
        public int Cols { get; }
        public float[,] Data { get; }

        public Matrix(int rows, int cols, bool fillZero = true)
        {
            Rows = rows;
            Cols = cols;
            Data = new float[rows, cols];

            if (!fillZero)
                Randomize();
        }

        public float this[int i, int j]
        {
            get => Data[i, j];
            set => Data[i, j] = value;
        }

        public void Randomize(float min = -1f, float max = 1f)
        {
            var rand = new Random();
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    Data[i, j] = (float)(rand.NextDouble() * (max - min) + min);
        }

        public Matrix Add(Matrix other)
        {
            if (Rows != other.Rows || Cols != other.Cols)
                throw new ArgumentException("Matrix dimensions must match for addition.");

            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[i, j] = this[i, j] + other[i, j];

            return result;
        }

        public Matrix Subtract(Matrix other)
        {
            if (Rows != other.Rows || Cols != other.Cols)
                throw new ArgumentException("Matrix dimensions must match for subtraction.");

            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[i, j] = this[i, j] - other[i, j];

            return result;
        }

        public Matrix Multiply(float scalar)
        {
            var result = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[i, j] = this[i, j] * scalar;

            return result;
        }

        public Matrix Dot(Matrix other)
        {
            if (Cols != other.Rows)
                throw new ArgumentException("Incompatible matrix dimensions for dot product.");

            var result = new Matrix(Rows, other.Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < other.Cols; j++)
                    for (int k = 0; k < Cols; k++)
                        result[i, j] += this[i, k] * other[k, j];

            return result;
        }

        public Matrix Transpose()
        {
            var result = new Matrix(Cols, Rows);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[j, i] = this[i, j];

            return result;
        }

        public Matrix Clone()
        {
            var clone = new Matrix(Rows, Cols);
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    clone[i, j] = this[i, j];
            return clone;
        }

        public void Print(string label = "Matrix")
        {
            Console.WriteLine($"\n{label} ({Rows}x{Cols}):");
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Cols; j++)
                    Console.Write($"{Data[i, j]:F2}\t");
                Console.WriteLine();
            }
        }
    }
}
