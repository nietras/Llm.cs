using System.Globalization;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Exporters;
using BenchmarkDotNet.Exporters.Csv;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Reports;
using Perfolizer.Horology;

namespace nietras.LargeLanguageModel.Benchmarks;

static class BenchmarkDotNetExtensions
{
    public static void ExportToFile(this IExporter exporter, Summary summary, string filePath)
    {
        using var logger = new StreamLogger(filePath);
        exporter.ExportToLog(summary, logger);
    }

}

class CustomMarkdownExporter : MarkdownExporter
{
    public CustomMarkdownExporter()
    {
        Dialect = "GitHub";
        UseCodeBlocks = true;
        CodeBlockStart = "```";
        StartOfGroupHighlightStrategy = MarkdownHighlightStrategy.None;
        ColumnsStartWithSeparator = true;
        EscapeHtml = true;
    }
}

class CustomCsvExporter : CsvExporter
{
    public CustomCsvExporter()
        : base(CsvSeparator.Semicolon, new SummaryStyle(
            cultureInfo: CultureInfo.InvariantCulture,
            printUnitsInHeader: true,
            printUnitsInContent: false,
            timeUnit: TimeUnit.Millisecond,
            sizeUnit: SizeUnit.KB
        ))
    { }
}

