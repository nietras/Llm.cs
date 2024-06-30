// Type 'Program' can be sealed because it has no subtypes in its containing assembly and is not externally visible
#pragma warning disable CA1852
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Environments;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using nietras.LargeLanguageModel.Benchmarks;
using nietras.SeparatedValues;
[assembly: System.Runtime.InteropServices.ComVisible(false)]

Action<string> log = t => { Console.WriteLine(t); Trace.WriteLine(t); };

log($"{Environment.Version} args: {args.Length}");

if (true || args.Length > 0)
{
    var name = nameof(Gpt2Bench);

    var markdownExporter = new CustomMarkdownExporter();
    var csvExporter = new CustomCsvExporter();
    var config = //(Debugger.IsAttached ? new DebugInProcessConfig() : DefaultConfig.Instance)
        ManualConfig.CreateEmpty()
            .AddColumnProvider(DefaultColumnProviders.Instance)
            .AddExporter(markdownExporter)
            .AddExporter(csvExporter)
            .AddLogger(ConsoleLogger.Default)
            .WithSummaryStyle(SummaryStyle.Default.WithMaxParameterColumnWidth(200));

    var summary = BenchmarkRunner.Run(typeof(Gpt2Bench), config, args);

    var cpuInfo = summary.HostEnvironmentInfo.CpuInfo.Value;
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

    var filePathMd = Path.Combine(directory, $"{name}.md");
    var filePathCsv = Path.Combine(directory, $"{name}.csv");

    markdownExporter.ExportToFile(summary, filePathMd);
    csvExporter.ExportToFile(summary, filePathCsv);

    var filePathBoard = Path.Combine(directory, $"{name}-Board.csv");

    UpdateBoardCsv(filePathCsv, filePathBoard);
}
else
{
    var b = new Gpt2Bench();
    b.GlobalSetup();
    b.Train();
    Thread.Sleep(200);
    for (var i = 0; i < 9; i++)
    {
        b.Train();
    }
    b.GlobalCleanup();
}

static string GetSourceDirectory([CallerFilePath] string filePath = "") =>
    Path.GetDirectoryName(filePath)!;

static void UpdateBoardCsv(string filePathCsv, string filePathBoard)
{
    const string colNameName = "Name";
    const string colNameMean = "Mean [ms]";

    string[] colNames = [colNameName, colNameMean, "StdDev [ms]", "Allocated [KB]"];

    var nameToCols = ReadNameToCols(filePathCsv, colNameName, colNameMean, colNames);
    if (File.Exists(filePathBoard))
    {
        var nameToColsBoard = ReadNameToCols(filePathBoard, colNameName, colNameMean, colNames);
        foreach (var (n, v) in nameToColsBoard)
        {
            if (!nameToCols.ContainsKey(n))
            {
                nameToCols[n] = v;
            }
        }
    }

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
