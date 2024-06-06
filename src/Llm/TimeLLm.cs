using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace nietras.LargeLanguageModel;

public unsafe class TimeLlm<TLlm>
    where TLlm : ILlm
{
    readonly SortedDictionary<TimeKey, List<long>> _keyToTimes = new();

    readonly record struct TimeKey(string Part, int Index, string CallerMemberName) : IComparable<TimeKey>
    {
        public int CompareTo(TimeKey other)
        {
            var c = Part.CompareTo(other.Part);
            if (c != 0) { return c; }
            return Index.CompareTo(other.Index);
        }
    }

    readonly ref struct Timer
    {
        readonly TimeLlm<TLlm> _llm;
        readonly string _callerMemberName;
        readonly long _start;

        public Timer(TimeLlm<TLlm> llm, string callerMemberName)
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
            }
            times.Add(time);
        }
    }

    public string Part { get; set; } = string.Empty;
    public int Index { get; set; } = -1;

    Timer NewTimer([CallerMemberName] string callerMemberName = "") =>
        new(this, callerMemberName);

    public void EmbedForward(
            int* tokenIndices, float* tokenEmbeddings, float* positionEmbeddings,
            int batchSize, int tokenCount, int channelCount,
            float* output)
    {
        using var _ = NewTimer();
        EmbedForward(tokenIndices, tokenEmbeddings, positionEmbeddings, batchSize, tokenCount, channelCount, output);
    }
    public void EmbedBackward(
            float* δoutput, int* tokenIndices,
            int batchSize, int tokenCount, int channelCount,
            float* δtokenEmbeddings, float* δpositionEmbeddings)
    {
        using var _ = NewTimer();
        EmbedBackward(δoutput, tokenIndices, batchSize, tokenCount, channelCount, δtokenEmbeddings, δpositionEmbeddings);
    }

    public void LayerNormForward(
            float* input, float* weight, float* bias, int batchSize, int tokenCount, int channelCount,
            float* mean, float* invStdDev, float* output)
    {
        using var _ = NewTimer();
    }
    public void LayerNormBackward(
            float* δoutput, float* input, float* weight, float* mean, float* invStdDev,
            int batchSize, int tokenCount, int channelCount,
            float* δweight, float* δbias, float* δinput)
    {
        using var _ = NewTimer();
    }

    public void MatMulForward(
            float* input, float* weight, float* bias,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* output)
    {
        using var _ = NewTimer();
    }
    public void MatMulBackward(
            float* δoutput, float* input, float* weight,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* δweight, float* δbias, float* δinput)
    {
        using var _ = NewTimer();
    }

    public void AttentionForward(
        float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* preAttention, float* postAttention, float* output)
    {
        using var _ = NewTimer();
    }
    public void AttentionBackward(
        float* δoutput, float* postAttention, float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* δpreAttention, float* δpostAttention, float* δinput)
    {
        using var _ = NewTimer();
    }

    public void GeLUForward(float* input, int count, float* output)
    {
        using var _ = NewTimer();
    }
    public void GeLUBackward(float* δoutput, float* input, int count, float* δinput)
    {
        using var _ = NewTimer();
    }

    public void ResidualForward(float* left, float* right, int count, float* output)
    {
        using var _ = NewTimer();
    }
    public void ResidualBackward(float* δoutput, int count, float* δleft, float* δright)
    {
        using var _ = NewTimer();
    }

    public void SoftmaxForward(float* logits,
        int batchSize, int tokenCount, int vocabularySize,
        float* probabilities)
    {
        using var _ = NewTimer();
    }

    public void CrossEntropyForward(
        float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* losses)
    {
        using var _ = NewTimer();
    }

    public void CrossEntropySoftmaxBackward(
        float* δlosses, float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* δlogits)
    {
        using var _ = NewTimer();
    }


    public void AdamW(
        float* gradients, float* ms, float* vs, float* parameters,
        long parameterCount, float learningRate,
        float beta1, float beta2, float eps, float weightDecay, int t)
    {
        using var _ = NewTimer();
    }
}
