using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

internal static partial class Gpt2
{
    public class ExpectedTokenTensors(int B, int T, object s) : Tensors<int>(s)
    {
        public static ExpectedTokenTensors Create(int batchSize, int tokenCount)
            => Create<ExpectedTokenTensors>(s => new(batchSize, tokenCount, s));

        public int BatchSize { get; } = B;
        public int TokenCount { get; } = T;
        public Tensor<int> InputTokenIndices { get; } = New([B, T], s);
        public Tensor<int> OutputTokenIndices { get; } = New([B, T], s);
    }

    public unsafe class ExpectedOutputTensors(int B, int T, int V, object s) : Tensors<float>(s)
    {
        public static ExpectedOutputTensors Create(int batchSize, int tokenCount, int vocabularySize)
            => Create<ExpectedOutputTensors>(s => new(batchSize, tokenCount, vocabularySize, s));

        public int BatchSize { get; } = B;
        public int TokenCount { get; } = T;
        public Tensor<float> ExpectedLogits { get; } = New([B, T, V], s);
    }

    public static unsafe void Test(string dataDirectory, ILlm llmToUse, int steps, Action<string>? log)
    {
        // build the GPT-2 model from a checkpoint
        using var model = ModelFromCheckpoint(dataDirectory + ModelBinaryFileName);
        int vocabularySize = model.Config.VocabularySize;
        int channelCount = model.Config.ChannelCount;
        int maxTokenCount = model.Config.MaxTokenCount;
        int layerCount = model.Config.LayerCount;

        var (expectedTokens, expectedOutputs, expectedGrads) = ReadExpectedState(model, dataDirectory);
        using (expectedTokens)
        using (expectedOutputs)
        using (expectedGrads)
        {
            log?.Invoke("[State]");
            log?.Invoke($"BatchSize: {expectedTokens.BatchSize}");
            log?.Invoke($"TokenCount: {expectedTokens.TokenCount}");

            // expected losses are as follows, from Python
            float[] expectedLosses = [
                5.270007133483887f,
                4.059706687927246f,
                3.3751230239868164f,
                2.8007826805114746f,
                2.315382242202759f,
                1.8490285873413086f,
                1.3946564197540283f,
                0.9991465210914612f,
                0.6240804195404053f,
                0.37651097774505615f
            ];

            // overall OK signal for the test
            bool allOk = true;

            // training iterations, following the pytorch code
            float* losses = stackalloc float[steps];
            var llm = CreateTimeLlm(llmToUse);
            for (int step = 0; step < steps; step++)
            {
                var timingEnabled = step >= JitAndWarmupCount;
                llm.Enabled = timingEnabled;

                var (loss, t) = TrainStep(model,
                    expectedTokens.InputTokenIndices, expectedTokens.OutputTokenIndices,
                    expectedTokens.BatchSize, expectedTokens.TokenCount,
                    llm, step);

                // error checking at step 0 for reference activations/gradients
                if (step == 0)
                {
                    // at this point
                    var logitsOk = CheckTensor(expectedOutputs.ExpectedLogits, model.Outputs!.Logits,
                        expectedOutputs.BatchSize * expectedOutputs.TokenCount * vocabularySize, "Logits");

                    var gradsOk = CheckTensors(model.ParameterGradients!, expectedGrads, "δ");
                    allOk &= logitsOk && gradsOk;
                }
                losses[step] = loss;
                var warmupMessage = timingEnabled ? "" : " JIT/WARMUP";
                if (step < expectedLosses.Length)
                {
                    var expectedLoss = expectedLosses[step];
                    var lossOk = CheckLoss(loss, expectedLoss);
                    allOk = allOk && lossOk;
                    log?.Invoke($"{step,2}: loss {loss:F6} exp. {expectedLoss:F6} {(lossOk ? "OK" : "FAIL"),-4} " +
                                $"({t.ToReport()}){warmupMessage}");
                }
                else
                {
                    log?.Invoke($"{step,2}: loss {loss:F6} ({t.ToReport()}){warmupMessage}");
                }
            }
            log?.Invoke($"All okay: {allOk}");

            var timeReport = llm.CreateReport(steps - JitAndWarmupCount);

            log?.Invoke(timeReport);

            if (!allOk) { throw new ArithmeticException($"{llmToUse.GetType().Name} failed {nameof(Gpt2)} train test run, see output for details."); }
        }
    }

    internal static unsafe
        (ExpectedTokenTensors ExpectedTokens,
         ExpectedOutputTensors ExpectedOutputs,
         ParameterTensors ExpectedGrads)
        ReadExpectedState(in Model model, string dataDirectory)
    {
        using var stateFile = File.OpenRead(dataDirectory + ModelDebugBinaryFileName);

        var (expectedTokens, expectedOutputs) = ReadExpectedTensors(model.Config.VocabularySize, stateFile);

        var expectedGrads = ParameterTensors.Create(model.Config);
        stateFile.ReadExactlyUnmanaged(expectedGrads.MemoryPtr, expectedGrads.TotalCount);

        return (expectedTokens, expectedOutputs, expectedGrads);
    }

    internal static unsafe (ExpectedTokenTensors ExpectedTokens,
         ExpectedOutputTensors ExpectedOutputs) ReadExpectedTensors(int vocabularySize, FileStream stateFile)
    {
        Span<int> stateHeader = stackalloc int[256];
        stateFile.ReadExactlyUnmanaged(stateHeader);

        if (stateHeader[0] != 20240327) { throw new InvalidDataException($"Bad magic model file"); }
        if (stateHeader[1] != 1) { throw new InvalidDataException($"Bad version in model file"); }
        int batchSize = stateHeader[2]; // batch size, e.g. 4
        int tokenCount = stateHeader[3]; // time/tokenCount / sequence length (e.g. 64, up to maxT)

        var expectedTokens = ExpectedTokenTensors.Create(batchSize, tokenCount);
        var expectedOutputs = ExpectedOutputTensors.Create(batchSize, tokenCount, vocabularySize);
        var expectedLoss = -1f;

        stateFile.ReadExactlyUnmanaged(expectedTokens.MemoryPtr, expectedTokens.TotalCount);
        stateFile.ReadExactlyUnmanaged(expectedOutputs.MemoryPtr, expectedOutputs.TotalCount);
        stateFile.ReadExactlyUnmanaged(&expectedLoss, 1);
        // Not using loss since already known separately

        return (expectedTokens, expectedOutputs);
    }

    static unsafe bool CheckTensors(
        IReadOnlyList<Tensor<float>> actuals,
        IReadOnlyList<Tensor<float>> expecteds,
        string namePrefix)
    {
        var allOk = actuals.Count == expecteds.Count;
        for (int i = 0; i < actuals.Count; i++)
        {
            var a = actuals[i];
            var e = expecteds[i];
            Debug.Assert(a.Name == e.Name);
            Debug.Assert(a.Count == e.Count);
            var ok = CheckTensor(a.Ptr, e.Ptr, a.Count, namePrefix + a.Name);
            allOk = allOk && ok;
        }
        return allOk;
    }

    const float CheckDiffLimit = 0.01f;
    static bool CheckLoss(float a, float b) => Check(a, b);
    static bool Check(float a, float b) => MathF.Abs(a - b) < CheckDiffLimit;

    // poor man's tensor checker
    static unsafe bool CheckTensor(float* actual, float* expected, nint count, string label)
    {
        const int printUpTo = 0;//5;
        LogNoNewLine($"{label,-30} ");
        bool ok = true;
        var maxAbsDiff = 0f;
        for (nint i = 0; i < count; i++)
        {
            var a = actual[i];
            var e = expected[i];

            var absDiff = MathF.Abs(a - e);
            maxAbsDiff = MathF.Max(absDiff, maxAbsDiff);

            var isOk = absDiff < CheckDiffLimit;
            ok &= isOk;
            if (i < printUpTo)
            {
                Log("");
                LogNoNewLine($"{(isOk ? "OK  " : "FAIL")} {a,15} {e,15} {absDiff,15}");
            }
            if (!isOk) { Debugger.Break(); }
        }
        Log($"TENSOR {(ok ? "OK  " : "FAIL")} MaxAbsDiff {maxAbsDiff,8:F6}");
        return ok;
    }
}
