// Type 'Program' can be sealed because it has no subtypes in its containing assembly and is not externally visible
#pragma warning disable CA1852
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Reflection;
using System.Runtime.CompilerServices;
using BenchmarkDotNet.Environments;
using nietras.LargeLanguageModel;
using nietras.SeparatedValues;
[assembly: System.Runtime.InteropServices.ComVisible(false)]

Action<string> log = t => { Console.WriteLine(t); Trace.WriteLine(t); };

var location = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
var dataDirectory = Path.Combine(location!, "../../../");

log($"{Environment.Version} args: {args.Length}");

// Download the model and tokenizer files if they don't exist
DownloadBinaryFilesIfNotExists(Gpt2.FileNames, Gpt2.RemoteUrl, dataDirectory, log);

var name = args?.Length > 0 ? args[0] : LlmFactory.DefaultName;
var llm = LlmFactory.NameToCreate[name]();

// Log to file too for reference
var logFilePath = Path.Combine(dataDirectory, $"{name}.log");
using var logWriter = new StreamWriter(logFilePath);
Action<string> newLog = t => { log(t); logWriter.WriteLine(t); };

//Gpt2.Infer(dataDirectory, llm, newLog);
const int steps = 10;
var meanStep_ms = Gpt2.VerifyTrain(dataDirectory, llm, steps, newLog);
//Gpt2.Train(dataDirectory, llm);
var boardName = nameof(Gpt2.VerifyTrain);

var cpuInfo = HostEnvironmentInfo.GetCurrent().CpuInfo.Value;
var processorName = ProcessorBrandStringHelper.Prettify(cpuInfo);
var processorNameInDirectory = processorName
    .Replace(" Processor", "").Replace(" CPU", "")
    .Replace(" Graphics", "")
    .Replace("/", "").Replace("\\", "")
    .Replace(" ", ".");
log(processorName);

//var processorNameInDirectory = "AMD.Ryzen.7.PRO.7840U.w.Radeon.780M";

var sourceDirectory = GetSourceDirectory();
var benchmarksDirectory = $"{sourceDirectory}/../../benchmarks/";
var directory = $"{benchmarksDirectory}{processorNameInDirectory}";
if (!Directory.Exists(directory)) { Directory.CreateDirectory(directory); }

var filePathBoard = Path.Combine(directory, $"{boardName}-Board.csv");

UpdateBoardCsv(name, meanStep_ms, filePathBoard);

static void DownloadBinaryFilesIfNotExists(
    IReadOnlyList<string> fileNames, Func<string, string> toUrl,
    string dataDirectory, Action<string>? log)
{
    foreach (var fileName in fileNames)
    {
        var filePath = Path.Combine(dataDirectory, fileName);
        filePath = Path.GetFullPath(filePath);
        if (!File.Exists(filePath))
        {
            var url = toUrl(fileName);
            log?.Invoke($"Downloading '{url}' to '{filePath}'");
            using var client = new HttpClient();
            // Download the file
            var source = client.GetStreamAsync(url).Result;
            using var destination = new FileStream(filePath, FileMode.Create);
            source.CopyTo(destination);
        }
    }
}

static string GetSourceDirectory([CallerFilePath] string filePath = "") =>
    Path.GetDirectoryName(filePath)!;

static void UpdateBoardCsv(string name, double mean_ms, string filePathBoard)
{
    const string colNameName = "Name";
    const string colNameMean = "Mean [ms]";

    string[] colNames = [colNameName, colNameMean]; //, "StdDev [ms]", "Allocated [KB]"];

    var value = (mean_ms, (string[])[name, mean_ms.ToString("F0")]);

    //if ()
    var nameToCols = File.Exists(filePathBoard)
        ? ReadNameToCols(filePathBoard, colNameName, colNameMean, colNames)
        : new() { { name, value } };
    //var nameToCols = ReadNameToCols(filePathCsv, colNameName, colNameMean, colNames);
    //if (File.Exists(filePathBoard))
    //{
    //    var nameToColsBoard = ReadNameToCols(filePathBoard, colNameName, colNameMean, colNames);
    //    foreach (var (n, v) in nameToColsBoard)
    //    {
    //        if (!nameToCols.ContainsKey(n))
    //        {
    //            nameToCols[n] = v;
    //        }
    //    }
    //}

    using var writerBoard = Sep.Writer().ToFile(filePathBoard);
    var sorted = nameToCols.Values.OrderBy(v => v.Mean);
    foreach (var (_, cols) in sorted)
    {
        using var writeRow = writerBoard.NewRow();
        writeRow[colNames].Set(cols);
    }
}

static Dictionary<string, (double Mean, string[] Cols)> ReadNameToCols(
    string filePath, string colNameName, string colNameMean, string[] colNames)
{
    using var reader = Sep
        .Reader(o => o with { Unescape = true, DisableFastFloat = true })
        .FromFile(filePath);
    return reader.Enumerate(r => (Name: r[colNameName].ToString(),
            Mean: r[colNameMean].Parse<double>(), Cols: r[colNames].ToStringsArray()))
        .ToDictionary(t => t.Name, t => (t.Mean, t.Cols));
}
