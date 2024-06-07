using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;

namespace nietras.LargeLanguageModel;

internal unsafe class TimeLlm(ILlm llm)
{
    readonly SortedDictionary<TimeKey, List<long>> _keyToTimes = [];
    readonly ILlm _llm = llm;

    public string Part { get; set; } = string.Empty;
    public int Index { get; set; } = -1;

    public Timer NewTimer([CallerMemberName] string callerMemberName = "") =>
        new(this, callerMemberName);

    public void EmbedForward(
            int* tokenIndices, float* tokenEmbeddings, float* positionEmbeddings,
            int batchSize, int tokenCount, int channelCount,
            float* output)
    {
        using var _ = NewTimer();
        _llm.EmbedForward(tokenIndices, tokenEmbeddings, positionEmbeddings, batchSize, tokenCount, channelCount, output);
    }
    public void EmbedBackward(
            float* δoutput, int* tokenIndices,
            int batchSize, int tokenCount, int channelCount,
            float* δtokenEmbeddings, float* δpositionEmbeddings)
    {
        using var _ = NewTimer();
        _llm.EmbedBackward(δoutput, tokenIndices, batchSize, tokenCount, channelCount, δtokenEmbeddings, δpositionEmbeddings);
    }

    public void LayerNormForward(
            float* input, float* weight, float* bias, int batchSize, int tokenCount, int channelCount,
            float* mean, float* invStdDev, float* output)
    {
        using var _ = NewTimer();
        _llm.LayerNormForward(input, weight, bias, batchSize, tokenCount, channelCount, mean, invStdDev, output);
    }
    public void LayerNormBackward(
            float* δoutput, float* input, float* weight, float* mean, float* invStdDev,
            int batchSize, int tokenCount, int channelCount,
            float* δweight, float* δbias, float* δinput)
    {
        using var _ = NewTimer();
        _llm.LayerNormBackward(δoutput, input, weight, mean, invStdDev, batchSize, tokenCount, channelCount, δweight, δbias, δinput);
    }

    public void MatMulForward(
            float* input, float* weight, float* bias,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* output)
    {
        using var _ = NewTimer();
        _llm.MatMulForward(input, weight, bias, batchSize, tokenCount, inputChannelCount, outputChannelCount, output);
    }
    public void MatMulBackward(
            float* δoutput, float* input, float* weight,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* δweight, float* δbias, float* δinput)
    {
        using var _ = NewTimer();
        _llm.MatMulBackward(δoutput, input, weight, batchSize, tokenCount, inputChannelCount, outputChannelCount, δweight, δbias, δinput);
    }

    public void AttentionForward(
        float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* preAttention, float* postAttention, float* output)
    {
        using var _ = NewTimer();
        _llm.AttentionForward(input, batchSize, tokenCount, channelCount, headCount, preAttention, postAttention, output);
    }
    public void AttentionBackward(
        float* δoutput, float* postAttention, float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* δpreAttention, float* δpostAttention, float* δinput)
    {
        using var _ = NewTimer();
        _llm.AttentionBackward(δoutput, postAttention, input, batchSize, tokenCount, channelCount, headCount, δpreAttention, δpostAttention, δinput);
    }

    public void GeLUForward(float* input, int count, float* output)
    {
        using var _ = NewTimer();
        _llm.GeLUForward(input, count, output);
    }
    public void GeLUBackward(float* δoutput, float* input, int count, float* δinput)
    {
        using var _ = NewTimer();
        _llm.GeLUBackward(δoutput, input, count, δinput);
    }

    public void ResidualForward(float* left, float* right, int count, float* output)
    {
        using var _ = NewTimer();
        _llm.ResidualForward(left, right, count, output);
    }
    public void ResidualBackward(float* δoutput, int count, float* δleft, float* δright)
    {
        using var _ = NewTimer();
        _llm.ResidualBackward(δoutput, count, δleft, δright);
    }

    public void SoftmaxForward(float* logits,
        int batchSize, int tokenCount, int vocabularySize,
        float* probabilities)
    {
        using var _ = NewTimer();
        _llm.SoftmaxForward(logits, batchSize, tokenCount, vocabularySize, probabilities);
    }

    public void CrossEntropyForward(
        float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* losses)
    {
        using var _ = NewTimer();
        _llm.CrossEntropyForward(probabilities, targetTokenIndices, batchSize, tokenCount, vocabularySize, losses);
    }

    public void CrossEntropySoftmaxBackward(
        float* δlosses, float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* δlogits)
    {
        using var _ = NewTimer();
        _llm.CrossEntropySoftmaxBackward(δlosses, probabilities, targetTokenIndices, batchSize, tokenCount, vocabularySize, δlogits);
    }

    public void AdamW(
        float* gradients, float* ms, float* vs, float* parameters,
        long parameterCount, float learningRate,
        float beta1, float beta2, float eps, float weightDecay, int t)
    {
        using var _ = NewTimer();
        _llm.AdamW(gradients, ms, vs, parameters, parameterCount, learningRate, beta1, beta2, eps, weightDecay, t);
    }

    public void Zero(float* output, long count)
    {
        using var _ = NewTimer();
        Gpt2.memset(output, count);
    }

    internal void Trace(Action<string> log)
    {
        var keyToStats = _keyToTimes.ToDictionary(p => p.Key, p => ComputeStats(p.Value));
        var totalSum_ms = keyToStats.Values.Sum(s => s.Sum_ms);

        var phase = string.Empty;
        foreach (var (key, stats) in keyToStats)
        {
            if (phase != key.Phase) { log(string.Empty); }
            phase = key.Phase;

            log($"{key.Phase,-10} {key.Index:D2} {key.CallerMemberName,-27} " +
                $"{stats.Sum_ms / totalSum_ms,3:P0} count: {stats.Count,3} sum: {stats.Sum_ms,6:F1} " +
                $"min: {stats.Min_ms,5:F1} mean: {stats.Mean_ms,5:F1} max: {stats.Max_ms,5:F1} [ms]");
        }

        log("");

        var methodToSum = keyToStats.GroupBy(p => p.Key.CallerMemberName)
                                    .Select(g => (Method: g.Key, Sum_ms: g.Sum(s => s.Value.Sum_ms)))
                                    .OrderByDescending(t => t.Sum_ms)
                                    .ToList();
        methodToSum.Add(("Total", totalSum_ms));
        foreach (var (method, sum_ms) in methodToSum)
        {
            log($"{method,-27} {sum_ms / totalSum_ms,4:P0} sum: {sum_ms,6:F0} [ms]");
        }
    }

    static TimeStats ComputeStats(List<long> times)
    {
        var min = long.MaxValue;
        var max = long.MinValue;
        long sum = 0;
        foreach (var time in times)
        {
            min = Math.Min(min, time);
            max = Math.Max(max, time);
            sum += time;
        }
        var mean = sum / (double)times.Count;
        var toMs = 1000.0 / Stopwatch.Frequency;
        var sum_ms = sum * toMs;
        var min_ms = min * toMs;
        var mean_ms = mean * toMs;
        var max_ms = max * toMs;
        return new(times.Count, sum_ms, min_ms, mean_ms, max_ms);
    }

    readonly record struct TimeKey(string Phase, int Index, string CallerMemberName)
        : IComparable<TimeKey>
    {
        public int CompareTo(TimeKey other)
        {
            var c = Phase.CompareTo(other.Phase);
            if (c != 0) { return c; }
            return Index.CompareTo(other.Index);
        }
    }

    readonly record struct TimeStats(int Count, double Sum_ms,
        double Min_ms, double Mean_ms, double Max_ms);

    internal readonly ref struct Timer
    {
        readonly TimeLlm _llm;
        readonly string _callerMemberName;
        readonly long _start;

        public Timer(TimeLlm llm, string callerMemberName)
        {
            _llm = llm;
            _callerMemberName = callerMemberName;
            ++_llm.Index;
            _start = Stopwatch.GetTimestamp();
        }

        public void Dispose()
        {
            var end = Stopwatch.GetTimestamp();
            var time = end - _start;
            TimeKey key = new(_llm.Part, _llm.Index, _callerMemberName);
            if (!_llm._keyToTimes.TryGetValue(key, out var times))
            {
                times = [];
                _llm._keyToTimes.Add(key, times);
            }
            times.Add(time);
        }
    }
}
