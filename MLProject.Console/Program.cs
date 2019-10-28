using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLProject.ConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<ImageNetData>(@"C:\Users\Emrek\source\repos\MLProject\MLProject.Console\images\tags.tsv", hasHeader: false);

            var pipeline = mlContext.Transforms

    // step 1: load the images
    .LoadImages(
        outputColumnName: "input",
        imageFolder: "images",
        inputColumnName: nameof(ImageNetData.ImagePath))

    // step 2: resize the images to 224x224
    .Append(mlContext.Transforms.ResizeImages(
        outputColumnName: "input",
        imageWidth: 224,
        imageHeight: 224,
        inputColumnName: "input"))

    // step 3: extract pixels in a format the TF model can understand
    // these interleave and offset values are identical to the images the model was trained on
    .Append(mlContext.Transforms.ExtractPixels(
        outputColumnName: "input",
        interleavePixelColors: true,
        offsetImage: 117))

    // step 4: load the TensorFlow model
    .Append(mlContext.Model.LoadTensorFlowModel(@"C:\Users\Emrek\source\repos\MLProject\MLProject.Console\models\tensorflow_inception_graph.pb")

    // step 5: score the images using the TF model
    .ScoreTensorFlowModel(
        outputColumnNames: new[] { "softmax2" },
        inputColumnNames: new[] { "input" },
        addBatchDimensionInput: true));

            Console.WriteLine("Start Training Model......");
            var model = pipeline.Fit(data);
            Console.WriteLine("Training was completed");

        

            var engine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

            var labels = File.ReadAllLines(@"C:\Users\Emrek\source\repos\MLProject\MLProject.Console\models\imagenet_comp_graph_label_strings");
            Console.WriteLine("Predicting....");
            var images = ImageNetData.ReadFromCsv(@"C:\Users\Emrek\source\repos\MLProject\MLProject.Console\images\tags.tsv");
            
            foreach (var image in images)
            {
                Console.WriteLine($"[{image.ImagePath}]:");
                float[] prediction = engine.Predict(image).PredictedLabels;

                var i = 0;
                var best = (from p in prediction
                            select new { Index = i++, Prediction = p }).OrderByDescending(p => p.Prediction).First();
                var predictedLabel = labels[best.Index];

                Console.WriteLine($"{predictedLabel} {(predictedLabel != image.Label ? "***WRONG***" : "")}");
            }

        }
    }

    public class ImageNetData
    {
        [LoadColumn(0)] public string ImagePath;
        [LoadColumn(1)] public string Label;


        public static IEnumerable<ImageNetData> ReadFromCsv(string file)
        {
            return File.ReadAllLines(file).Select(x => x.Split('\t')).Select(x => new ImageNetData
            {
                ImagePath = x[0],
                Label = x[1]
            }); 
        }
    }

    public class ImageNetPrediction
    {
        [ColumnName("softmax2")]
        public float[] PredictedLabels;
    }
}
