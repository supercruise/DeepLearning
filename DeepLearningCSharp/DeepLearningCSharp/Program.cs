using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;

namespace DeepLearningCSharp
{
    class Program
    {
        static void Main(string[] args)
        {
            // Generate training data set.
            double w1 = 2.0;
            double w2 = -3.4;
            double bias = 4.2;
            int numberOfExample = 1000;

            DataGenerator dg = new DataGenerator(w1, w2, bias, numberOfExample);
            dg.GenerateInputAndLabel();

            List<double> inputX1 = dg.GeneratedX1.ToList();
            List<double> inputX2 = dg.GeneratedX2.ToList();
            List<double> labelY = dg.GeneratedLabelY.ToList();

            // Train the neural network.
            LinearRegressionNN lrnn = new LinearRegressionNN(inputX1, inputX2, labelY);
            lrnn.TrainModel();

            // Generate testing data set.
            dg.Clear();
            dg.GenerateInputAndLabel();

            List<double> testX1 = dg.GeneratedX1.ToList();
            List<double> testX2 = dg.GeneratedX2.ToList();
            List<double> testY = dg.GeneratedLabelY.ToList();

            // Test the model
            for (int i = 0; i < 5; ++i)
            {
                double predictedY = lrnn.Predict(testX1[i], testX2[i]);
                double linReg = DLHelper.CalcLinReg(testX1[i], testX2[i], w1, w2, bias);

                //Console.WriteLine(string.Format("The real house price is      {0:f6}", testY[i]));
                Console.WriteLine(string.Format("The linear reg price is      {0:f6}", linReg));
                Console.WriteLine(string.Format("The predicted house price is {0:f6}", predictedY));
                Console.WriteLine();
            }

            Console.ReadLine();
        }
    }

    public class LinearRegressionNN
    {
        protected double trainedW1;
        protected double trainedW2;
        protected double trainedBias;
        protected List<double> inputX1 = new List<double>();
        protected List<double> inputX2 = new List<double>();
        protected List<double> labelY = new List<double>();
        protected List<double> outputY = new List<double>();
        protected double pace = 0.03;
        protected int numInput;

        public double TrainedW1
        {
            get { return trainedW1; }
        }

        public double TrainedW2
        {
            get { return trainedW2; }
        }

        public double TrainedBias
        {
            get { return trainedBias; }
        }

        public LinearRegressionNN(List<double> X1, List<double> X2, List<double> Y)
        {
            if (X1 == null || X2 == null || Y == null)
            {
                Debug.Assert(false);
            }

            if (X1.Count != X2.Count || X2.Count != Y.Count)
            {
                Debug.Assert(false);
            }

            numInput = X1.Count;

            inputX1 = X1;
            inputX2 = X2;
            labelY = Y;
        }

        protected double gradientDescentW(double x, double diff)
        {
            return -1 * pace * x * diff;
        }

        protected double gradientDescentB(double diff)
        {
            return -1 * pace * diff;
        }

        public void TrainModel()
        {
            double w1 = 0.1;
            double w2 = 0.1;
            double bias = 0.0;

            // Conduct gradient descent
            for (int j = 0; j < numInput; ++j)
            {
                double yHat = DLHelper.CalcLinReg(inputX1[j], inputX2[j], w1, w2, bias);
                double diff = yHat - labelY[j];

                // update w1, w2, and bias
                w1 += gradientDescentW(inputX1[j], diff);
                w2 += gradientDescentW(inputX2[j], diff);
                bias += gradientDescentB(diff);
            }

            trainedW1 = w1;
            trainedW2 = w2;
            trainedBias = bias;

            Console.WriteLine(string.Format("The trained W1 is {0:f2}", trainedW1));
            Console.WriteLine(string.Format("The trained W2 is {0:f2}", trainedW2));
            Console.WriteLine(string.Format("The trained bias is {0:f2}", trainedBias));
            Console.WriteLine();
        }

        public double Predict(double x1, double x2)
        {
            return trainedW1 * x1 + trainedW2 * x2 + trainedBias;
        }
    }

    public class DataGenerator
    {
        protected double trueW1;
        protected double trueW2;
        protected double trueBias;
        protected int numExamples = 1000;

        protected List<double> generatedX1 = new List<double>();
        protected List<double> generatedX2 = new List<double>();
        protected List<double> generatedLabelY = new List<double>();

        public DataGenerator(double w1, double w2, double bias, int number = 0)
        {
            trueW1 = w1;
            trueW2 = w2;
            trueBias = bias;
            numExamples = number;
        }

        public IReadOnlyList<double> GeneratedX1
        {
            get { return generatedX1.AsReadOnly(); }
        }

        public IReadOnlyList<double> GeneratedX2
        {
            get { return generatedX2.AsReadOnly(); }
        }

        public IReadOnlyList<double> GeneratedLabelY
        {
            get { return generatedLabelY.AsReadOnly(); }
        }

        public void GenerateInputAndLabel()
        {
            GenerateX1();
            GenerateX2();
            GenerateLabelY();
        }

        public void Clear()
        {
            generatedX1.Clear();
            generatedX2.Clear();
            generatedLabelY.Clear();
        }

        protected void GenerateX1()
        {
            double mean = 0.0;
            double stdDev = 1.0;
            Normal normalDist = new Normal(mean, stdDev);

            for (int i = 0; i < numExamples; ++i)
            {
                generatedX1.Add(normalDist.Sample());
            }
        }

        protected void GenerateX2()
        {
            double mean = 0.0;
            double stdDev = 1.0;
            Normal normalDist = new Normal(mean, stdDev);

            for (int i = 0; i < numExamples; ++i)
            {
                generatedX2.Add(normalDist.Sample());
            }
        }

        protected void GenerateLabelY()
        {
            if (generatedX1.Count == 0)
                GenerateX1();

            if (generatedX2.Count == 0)
                GenerateX2();

            double mean = 0.0;
            double stdDev = 0.01;
            Normal normalDist = new Normal(mean, stdDev);

            for (int i = 0; i < numExamples; ++i)
            {
                double y = DLHelper.CalcLinReg(generatedX1[i], generatedX2[i], trueW1, trueW2, trueBias) + normalDist.Sample();
                generatedLabelY.Add(y);
            }
        }
    }

    public class DLHelper
    {
        public static double CalcLinReg(double x1, double x2, double w1, double w2, double bias)
        {
            return x1 * w1 + x2 * w2 + bias;
        }

        public static double CalcSquareLoss(double yHat, double y)
        {
            return 0.5 * (yHat - y) * (yHat - y);
        }
    }
}
