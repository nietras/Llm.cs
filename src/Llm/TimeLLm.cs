using System.Runtime.CompilerServices;

namespace nietras.LargeLanguageModel;

public unsafe class TimeLlm<TLlm>
    where TLlm : ILlm
{
    readonly ref struct Time
    {
        public Time(string callerMemberName)
        {

        }

        public void Dispose()
        {
        }
    }

    Time Get([CallerMemberName] string callerMemberName = "")
    {
        return new(callerMemberName);
    }

    public void EmbedForward(
            int* tokenIndices, float* tokenEmbeddings, float* positionEmbeddings,
            int batchSize, int tokenCount, int channelCount,
            float* output)
    { }
    public void EmbedBackward(
            float* δoutput, int* tokenIndices,
            int batchSize, int tokenCount, int channelCount,
            float* δtokenEmbeddings, float* δpositionEmbeddings)
    { }

    public void LayerNormForward(
            float* input, float* weight, float* bias, int batchSize, int tokenCount, int channelCount,
            float* mean, float* invStdDev, float* output)
    { }
    public void LayerNormBackward(
            float* δoutput, float* input, float* weight, float* mean, float* invStdDev,
            int batchSize, int tokenCount, int channelCount,
            float* δweight, float* δbias, float* δinput)
    { }

    public void MatMulForward(
            float* input, float* weight, float* bias,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* output)
    { }
    public void MatMulBackward(
            float* δoutput, float* input, float* weight,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* δweight, float* δbias, float* δinput)
    { }

    public void AttentionForward(
        float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* preAttention, float* postAttention, float* output)
    { }
    public void AttentionBackward(
        float* δoutput, float* postAttention, float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* δpreAttention, float* δpostAttention, float* δinput)
    { }

    public void GeLUForward(float* input, int count, float* output) { }
    public void GeLUBackward(float* δoutput, float* input, int count, float* δinput) { }

    public void ResidualForward(float* left, float* right, int count, float* output) { }
    public void ResidualBackward(float* δoutput, int count, float* δleft, float* δright) { }

    public void SoftmaxForward(float* logits,
        int batchSize, int tokenCount, int vocabularySize,
        float* probabilities)
    { }
    public void CrossEntropyForward(
        float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* losses)
    { }
    public void CrossEntropySoftmaxBackward(
        float* δlosses, float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* δlogits)
    { }

    public void AdamW(
        float* gradients, float* ms, float* vs, float* parameters,
        long parameterCount, float learningRate,
        float beta1, float beta2, float eps, float weightDecay, int t)
    { }
}
