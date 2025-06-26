using CnnFromScratch.Core;
using NUnit.Framework;

namespace CnnFromScratch.Tests
{
    public class MatrixTests
    {
        [Test]
        public void Clone_CreatesDeepCopy()
        {
            // Arrange
            var original = new Matrix(2, 2);
            original[0, 0] = 1.0f;
            original[0, 1] = 2.0f;
            original[1, 0] = 3.0f;
            original[1, 1] = 4.0f;

            // Act
            var clone = original.Clone();
            
            // Modify original to verify deep copy
            original[0, 0] = 5.0f;

            // Assert
            Assert.Multiple(() =>
            {
                // Verify dimensions match
                Assert.That(clone.Rows, Is.EqualTo(original.Rows));
                Assert.That(clone.Cols, Is.EqualTo(original.Cols));

                // Verify contents were copied
                Assert.That(clone[0, 0], Is.EqualTo(1.0f)); // Original value, not modified
                Assert.That(clone[0, 1], Is.EqualTo(2.0f));
                Assert.That(clone[1, 0], Is.EqualTo(3.0f));
                Assert.That(clone[1, 1], Is.EqualTo(4.0f));

                // Verify it's a deep copy
                Assert.That(clone[0, 0], Is.Not.EqualTo(original[0, 0]));
            });
        }

        [Test]
        public void Clone_EmptyMatrix_CreatesEmptyCopy()
        {
            // Arrange
            var original = new Matrix(1, 1);

            // Act
            var clone = original.Clone();

            // Assert
            Assert.Multiple(() =>
            {
                Assert.That(clone.Rows, Is.EqualTo(1));
                Assert.That(clone.Cols, Is.EqualTo(1));
                Assert.That(clone[0, 0], Is.EqualTo(0.0f));
            });
        }
    }
}