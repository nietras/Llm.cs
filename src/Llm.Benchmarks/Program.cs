// Type 'Program' can be sealed because it has no subtypes in its containing assembly and is not externally visible
#pragma warning disable CA1852
using System;
using System.Diagnostics;
using System.Threading;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using nietras.LargeLanguageModel.Benchmarks;
[assembly: System.Runtime.InteropServices.ComVisible(false)]

Action<string> log = t => { Console.WriteLine(t); Trace.WriteLine(t); };

log($"{Environment.Version} args: {args.Length}");

if (args.Length > 0)
{
    var config = (Debugger.IsAttached ? new DebugInProcessConfig() : DefaultConfig.Instance)
        .WithSummaryStyle(SummaryStyle.Default.WithMaxParameterColumnWidth(200))
        //.AddColumn(MBPerSecFromCharsLength())
        ;
    BenchmarkRunner.Run(typeof(LlmBench), config, args);
}
else
{
    var b = new LlmBench();
    b.GlobalSetup();
    b.Naive();
    Thread.Sleep(200);
    for (var i = 0; i < 200000000; i++)
    {
        b.Naive();
    }
}
