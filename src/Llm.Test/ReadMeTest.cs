using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
#if NET8_0
using PublicApiGenerator;
#endif
#pragma warning disable CA2007 // Consider calling ConfigureAwait on the awaited task

// Only parallize on class level to avoid multiple writes to README file
[assembly: Parallelize(Workers = 0, Scope = ExecutionScope.ClassLevel)]

namespace nietras.LargeLanguageModel.Test;

[TestClass]
public class ReadMeTest
{
    static readonly string s_testSourceFilePath = SourceFile();
    static readonly string s_rootDirectory = Path.GetDirectoryName(s_testSourceFilePath) + @"../../../";
    static readonly string s_readmeFilePath = s_rootDirectory + @"README.md";

    [TestMethod]
    public void ReadMeTest_()
    {

    }

    [TestMethod]
    public void ReadMeTest_UpdateBenchmarksInMarkdown()
    {
        var readmeFilePath = s_readmeFilePath;

        var benchmarkFileNameToConfig = new Dictionary<string, (string Description, string ReadmeBefore, string ReadmeEnd, string SectionPrefix)>()
        {
            //{ "PackageAssetsBench.md", new("PackageAssets Benchmark Results", "##### PackageAssets Benchmark Results", "##### PackageAssets", "###### ") },
            //{ "PackageAssetsBench-GcServer.md", new("PackageAssets Benchmark Results (SERVER GC)", "##### PackageAssets Benchmark Results (SERVER GC)", "##### ", "###### ") },
            //{ "PackageAssetsBenchQuotes.md", new("PackageAssets with Quotes Benchmark Results", "##### PackageAssets with Quotes Benchmark Results", "##### PackageAssets", "###### ") },
            //{ "PackageAssetsBenchQuotes-GcServer.md", new("PackageAssets with Quotes Benchmark Results (SERVER GC)", "##### PackageAssets with Quotes Benchmark Results (SERVER GC)", "#### ", "###### ") },
            //{ "FloatsReaderBench.md", new("FloatsReader Benchmark Results", "#### Floats Reader Comparison Benchmarks", "### Writer", "##### ") },
        };

        var benchmarksDirectory = Path.Combine(s_rootDirectory, "benchmarks");
        if (Directory.Exists(benchmarksDirectory))
        {
            var processorDirectories = Directory.EnumerateDirectories(benchmarksDirectory).ToArray();
            var processors = processorDirectories.Select(LastDirectoryName).ToArray();

            var readmeLines = File.ReadAllLines(readmeFilePath);

            foreach (var (fileName, config) in benchmarkFileNameToConfig)
            {
                var description = config.Description;
                var prefix = config.SectionPrefix;
                var readmeBefore = config.ReadmeBefore;
                var readmeEndLine = config.ReadmeEnd;
                var all = "";
                foreach (var processorDirectory in processorDirectories)
                {
                    var versions = File.ReadAllText(Path.Combine(processorDirectory, "Versions.txt"));
                    var contents = File.ReadAllText(Path.Combine(processorDirectory, fileName));
                    var processor = LastDirectoryName(processorDirectory);

                    var section = $"{prefix}{processor} - {description} ({versions})";
                    var benchmarkTable = GetBenchmarkTable(contents);
                    var readmeContents = $"{section}{Environment.NewLine}{Environment.NewLine}{benchmarkTable}{Environment.NewLine}";
                    all += readmeContents;
                }
                readmeLines = ReplaceReadmeLines(readmeLines, [all], readmeBefore, prefix, 0, readmeEndLine, 0);
            }

            var newReadme = string.Join(Environment.NewLine, readmeLines) + Environment.NewLine;
            File.WriteAllText(readmeFilePath, newReadme, Encoding.UTF8);
        }
        static string LastDirectoryName(string d) =>
            d.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar).Last();

        static string GetBenchmarkTable(string markdown) =>
            markdown.Substring(markdown.IndexOf('|'));
    }

    [TestMethod]
    public void ReadMeTest_UpdateExampleCodeInMarkdown()
    {
        var testSourceFilePath = s_testSourceFilePath;
        var readmeFilePath = s_readmeFilePath;
        var rootDirectory = s_rootDirectory;

        var readmeLines = File.ReadAllLines(readmeFilePath);

        // Update README examples
        var testSourceLines = File.ReadAllLines(testSourceFilePath);
        var testBlocksToUpdate = new (string StartLineContains, string ReadmeLineBeforeCodeBlock)[]
        {
            (nameof(ReadMeTest_) + "()", "## Example"),
            //(nameof(ReadMeTest_LlmReader_Debuggability) + "()", "#### LlmReader Debuggability"),
        };
        readmeLines = UpdateReadme(testSourceLines, readmeLines, testBlocksToUpdate,
            sourceStartLineOffset: 2, "    }", sourceEndLineOffset: 0, sourceWhitespaceToRemove: 8);

        var newReadme = string.Join(Environment.NewLine, readmeLines) + Environment.NewLine;
        File.WriteAllText(readmeFilePath, newReadme, Encoding.UTF8);
    }

    // Only update public API in README for .NET 8.0 to keep consistent
#if NET8_0
    [TestMethod]
    public void ReadMeTest_PublicApi()
    {
        var publicApi = typeof(Gpt2).Assembly.GeneratePublicApi();

        var readmeFilePath = s_readmeFilePath;
        var readmeLines = File.ReadAllLines(readmeFilePath);
        readmeLines = ReplaceReadmeLines(readmeLines, [publicApi],
            "## Public API Reference", "```csharp", 1, "```", 0);

        var newReadme = string.Join(Environment.NewLine, readmeLines) + Environment.NewLine;
        File.WriteAllText(readmeFilePath, newReadme, Encoding.UTF8);
    }
#endif

    static string[] UpdateReadme(string[] sourceLines, string[] readmeLines,
        (string StartLineContains, string ReadmeLineBefore)[] blocksToUpdate,
        int sourceStartLineOffset, string sourceEndLineStartsWith, int sourceEndLineOffset, int sourceWhitespaceToRemove,
        string readmeStartLineStartsWith = "```csharp", int readmeStartLineOffset = 1,
        string readmeEndLineStartsWith = "```", int readmeEndLineOffset = 0)
    {
        foreach (var (startLineContains, readmeLineBeforeBlock) in blocksToUpdate)
        {
            var sourceExampleLines = SnipLines(sourceLines,
                startLineContains, sourceStartLineOffset,
                sourceEndLineStartsWith, sourceEndLineOffset,
                sourceWhitespaceToRemove);

            readmeLines = ReplaceReadmeLines(readmeLines, sourceExampleLines, readmeLineBeforeBlock,
                readmeStartLineStartsWith, readmeStartLineOffset, readmeEndLineStartsWith, readmeEndLineOffset);
        }

        return readmeLines;
    }

    static string[] ReplaceReadmeLines(string[] readmeLines, string[] newReadmeLines, string readmeLineBeforeBlock,
        string readmeStartLineStartsWith, int readmeStartLineOffset,
        string readmeEndLineStartsWith, int readmeEndLineOffset)
    {
        var readmeLineBeforeIndex = Array.FindIndex(readmeLines,
            l => l.StartsWith(readmeLineBeforeBlock, StringComparison.Ordinal)) + 1;
        if (readmeLineBeforeIndex == 0)
        { throw new ArgumentException($"README line '{readmeLineBeforeBlock}' not found."); }

        return ReplaceReadmeLines(readmeLines, newReadmeLines,
            readmeLineBeforeIndex, readmeStartLineStartsWith, readmeStartLineOffset, readmeEndLineStartsWith, readmeEndLineOffset);
    }

    static string[] ReplaceReadmeLines(string[] readmeLines, string[] newReadmeLines, int readmeLineBeforeIndex,
        string readmeStartLineStartsWith, int readmeStartLineOffset,
        string readmeEndLineStartsWith, int readmeEndLineOffset)
    {
        var readmeReplaceStartIndex = Array.FindIndex(readmeLines, readmeLineBeforeIndex,
            l => l.StartsWith(readmeStartLineStartsWith, StringComparison.Ordinal)) + readmeStartLineOffset;
        var readmeReplaceEndIndex = Array.FindIndex(readmeLines, readmeReplaceStartIndex,
            l => l.StartsWith(readmeEndLineStartsWith, StringComparison.Ordinal)) + readmeEndLineOffset;

        readmeLines = readmeLines[..readmeReplaceStartIndex].AsEnumerable()
            .Concat(newReadmeLines)
            .Concat(readmeLines[readmeReplaceEndIndex..]).ToArray();
        return readmeLines;
    }

    static string[] SnipLines(string[] sourceLines,
        string startLineContains, int startLineOffset,
        string endLineStartsWith, int endLineOffset,
        int whitespaceToRemove = 8)
    {
        var sourceStartLine = Array.FindIndex(sourceLines,
            l => l.Contains(startLineContains, StringComparison.Ordinal));
        sourceStartLine += startLineOffset;
        var sourceEndLine = Array.FindIndex(sourceLines, sourceStartLine,
            l => l.StartsWith(endLineStartsWith, StringComparison.Ordinal));
        sourceEndLine += endLineOffset;
        var sourceExampleLines = sourceLines[sourceStartLine..sourceEndLine]
            .Select(l => l.Length > 0 ? l.Remove(0, whitespaceToRemove) : l).ToArray();
        return sourceExampleLines;
    }

    static string SourceFile([CallerFilePath] string sourceFilePath = "") => sourceFilePath;
}
