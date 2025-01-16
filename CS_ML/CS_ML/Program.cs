using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
/*
# AI Agent 範例 ~ copilot
有趣的挑戰啊！這裡有個簡單的範例，展示如何在 C# 使用 ML.NET 建立一個基本的機器學習代理。 

01.在 NuGet Package Manager 中搜索並安裝 Microsoft.ML

02.這是一個簡單的數據集範例：
csv
Size,SoldPrice
700,50000
800,60000
850,62500
將資料保存為名為 house_prices.csv 的文件。

03.在這個範例中，我們利用 ML.NET 訓練了一個簡單的線性回歸模型，並預測給定房屋面積（如 900 sq. ft.）的價格。
 */
public class HouseData
{
    [LoadColumn(0)]
    public float Size { get; set; }

    [LoadColumn(1)]
    public float SoldPrice { get; set; }
}


public class HousePricePrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        var dataPath = Path.Combine(Environment.CurrentDirectory, "house_prices.csv");
        var dataView = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');

        var pipeline = mlContext.Transforms.Concatenate("Features", new[] { "Size" })
                      .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "SoldPrice", maximumNumberOfIterations: 100));

        var model = pipeline.Fit(dataView);

        var size = new HouseData() { Size = 900 };//輸入
        var predictionFunction = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);
        var prediction = predictionFunction.Predict(size);

        Console.WriteLine($"預測房價: {prediction.Price}");
    }
}
