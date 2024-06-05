﻿namespace nietras.LargeLanguageModel;

public interface ILlm
{
    static abstract unsafe void EmbedForward(
            int* tokenIndices, float* tokenEmbeddings, float* positionEmbeddings,
            int batchSize, int tokenCount, int channelCount,
            float* output);
    static abstract unsafe void EmbedBackward(
            float* δoutput, int* tokenIndices,
            int batchSize, int tokenCount, int channelCount,
            float* δtokenEmbeddings, float* δpositionEmbeddings);

    static abstract unsafe void LayerNormForward(
            float* input, float* weight, float* bias, int batchSize, int tokenCount, int channelCount,
            float* mean, float* invStdDev, float* output);
    static abstract unsafe void LayerNormBackward(
            float* δoutput, float* input, float* weight, float* mean, float* invStdDev,
            int batchSize, int tokenCount, int channelCount,
            float* δweight, float* δbias, float* δinput);

    static abstract unsafe void MatMulForward(
            float* input, float* weight, float* bias,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* output);
    static abstract unsafe void MatMulBackward(
            float* δoutput, float* input, float* weight,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* δweight, float* δbias, float* δinput);

    static abstract unsafe void AttentionForward(
        float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* preAttention, float* postAttention, float* output);
    static abstract unsafe void AttentionBackward(
        float* δoutput, float* postAttention, float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* δpreAttention, float* δpostAttention, float* δinput);

    static abstract unsafe void GeLUForward(float* input, int count, float* output);
    static abstract unsafe void GeLUBackward(float* δoutput, float* input, int count, float* δinput);

    static abstract unsafe void ResidualForward(float* left, float* right, int count, float* output);
    static abstract unsafe void ResidualBackward(float* δoutput, int count, float* δleft, float* δright);

    static abstract unsafe void SoftmaxForward(float* logits,
        int batchSize, int tokenCount, int vocabularySize,
        float* probabilities);
    static abstract unsafe void CrossEntropyForward(
        float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* losses);
    static abstract unsafe void CrossEntropySoftmaxBackward(
        float* δlosses, float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* δlogits);
}