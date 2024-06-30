﻿using System.Collections.Generic;
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
    GPT2 _model = new();
    ExpectedTensors _expectedInputsOutputs;
    ParameterTensors _expectedGrads;
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
        BuildFromCheckpoint(ref _model, DataDirectory + ModelBinaryFileName);

        (_expectedInputsOutputs, _expectedGrads) = ReadExpectedState(_model, DataDirectory);

        var llm = LlmFactory.NameToCreate[Name]();
        _llm = new TimeLlm(llm);
        _step = 0;
    }

    [Benchmark]
    public unsafe float Train()
    {
        var (loss, t) = TrainStep(ref _model,
            _expectedInputsOutputs.InputTokenIndices, _expectedInputsOutputs.OutputTokenIndices,
            _expectedInputsOutputs.BatchSize, _expectedInputsOutputs.TokenCount,
            _llm!, _step);
        ++_step;
        return loss;
    }

    [GlobalCleanup]
    public unsafe void GlobalCleanup()
    {
        free(_expectedInputsOutputs.InputTokenIndices);
        free(_expectedInputsOutputs.OutputTokenIndices);
        free(_expectedInputsOutputs.ExpectedLogits);
        free(_expectedInputsOutputs.ExpectedLoss);
        free(_expectedGrads.MemoryPtr);
        Free(ref _model);
    }
}
