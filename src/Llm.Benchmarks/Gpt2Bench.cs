using System.Collections.Generic;
using System.Diagnostics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Order;
using static nietras.LargeLanguageModel.Gpt2;

namespace nietras.LargeLanguageModel.Benchmarks;

[MemoryDiagnoser]
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByMethod)]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
#if DEBUG
[WarmupCount(1)]
[MinIterationCount(2)]
[MaxIterationCount(3)]
#endif
public class Gpt2Bench
{
    const string DataDirectory = "../../../";
    Model _model = null!;
    ExpectedTokenTensors _expectedTokens = null!;
    ExpectedOutputTensors _expectedOutputs = null!;
    ParameterTensors _expectedGrads = null!;
    TimeLlm? _llm;
    int _step;

    [ParamsSource(nameof(NameParams))] // Attributes for params is challenging 👇
    public string Name { get; set; } = nameof(Llm);
    public static IEnumerable<string> NameParams() => LlmFactory.NameToCreate.Keys;

    [GlobalSetup]
    public void GlobalSetup()
    {
        Runner.DownloadBinaryFilesIfNotExists(Gpt2.FileNames, Gpt2.RemoteUrl,
            DataDirectory, t => Trace.WriteLine(t));

        // build the GPT-2 model from a checkpoint
        _model = ModelFromCheckpoint(DataDirectory + ModelBinaryFileName);

        (_expectedTokens, _expectedOutputs, _expectedGrads) =
            ReadExpectedState(_model, DataDirectory);

        var llm = LlmFactory.NameToCreate[Name]();
        _llm = new TimeLlm(llm);
        _step = 0;
    }

    [Benchmark]
    public unsafe float Train()
    {
        var (loss, t) = TrainStep(_model,
            _expectedTokens.InputTokenIndices, _expectedTokens.OutputTokenIndices,
            _expectedTokens.BatchSize, _expectedTokens.TokenCount,
            _llm!, _step);
        ++_step;
        return loss;
    }

    [GlobalCleanup]
    public unsafe void GlobalCleanup()
    {
        _expectedTokens.Dispose();
        _expectedOutputs.Dispose();
        _expectedGrads.Dispose();
        _model.Dispose();
    }
}
