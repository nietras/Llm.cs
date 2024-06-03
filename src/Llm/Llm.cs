using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using P = nietras.LargeLanguageModel.LlmParallel;
//using P = nietras.LargeLanguageModel.NotParallel;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

// all the individual layers' forward and backward passes
// batchSize = B, tokenCount = T, channelCount = C, vocabularySize = V
public partial class Llm : ILlm
{
    // Order of method parameters:
    // * Source memory
    // * Arguments
    // * Destination memory

    // δ (greek small letter delta) used for naming gradients/derivatives.
    // Perhaps nabla or math delta would be better but not allowed in C#
    // identifier.

    // Calling this "Encoder" is confusing as sounds like the entire other part
    // of transformer architecture so renamed to "Embed".

    internal Llm() { }

    /// <summary>
    /// Forward pass of the embedding layer.
    /// </summary>
    /// <param name="tokenIndices">Pointer to the input tensor of token indices/ids.</param>
    /// <param name="tokenEmbeddings">Pointer to the token embeddings tensor.</param>
    /// <param name="positionEmbeddings">Pointer to the position embeddings tensor.</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="channelCount">The number of channels.</param>
    /// <param name="output">Pointer to the output tensor.</param>
    public unsafe static void EmbedForward(
        // [batchSize, tokenCount], [vocabularySize, channelCount], [maxTokenCount, channelCount]
        int* tokenIndices, float* tokenEmbeddings, float* positionEmbeddings,
        int batchSize, int tokenCount, int channelCount,
        // [batchSize, tokenCount, channelCount]
        float* output)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                // seek to the output position in output[b, t, :]
                float* output_bt = output + b * tokenCount * channelCount + t * channelCount;
                // get the index of the token at input[b, t]
                int tokenIndex = tokenIndices[b * tokenCount + t];
                // seek to the position corresponding to the token [tokenIndex, :]
                float* tokenEmbeddingVector = tokenEmbeddings + tokenIndex * channelCount;
                // seek to the position corresponding to the position [t, :]
                float* positionEmbeddingVector = positionEmbeddings + t * channelCount;
                // add the two vectors and store the result in output[b, t ,:]
                for (int c = 0; c < channelCount; c++)
                {
                    output_bt[c] = tokenEmbeddingVector[c] + positionEmbeddingVector[c];
                }
            }
        }
    }

    /// <summary>
    /// Backward pass of the embedding layer.
    /// </summary>
    /// <param name="δoutput">Pointer to the output derivative tensor of shape [batchSize, tokenCount, channelCount].</param>
    /// <param name="tokenIndices">Pointer to the token indices tensor of shape [batchSize, tokenCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="channelCount">The number of channels.</param>
    /// <param name="δtokenEmbeddings">Pointer to the token embeddings derivative tensor of shape [vocabularySize, channelCount].</param>
    /// <param name="δpositionEmbeddings">Pointer to the position embeddings derivative tensor of shape [maxTokenCount, channelCount].</param>
    public unsafe static void EmbedBackward(
        // [batchSize, tokenCount, channelCount], [batchSize, tokenCount]
        float* δoutput, int* tokenIndices,
        int batchSize, int tokenCount, int channelCount,
        // [vocabularySize, channelCount], [maxTokenCount, channelCount]
        float* δtokenEmbeddings, float* δpositionEmbeddings)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                float* outputGrad_bt = δoutput + b * tokenCount * channelCount + t * channelCount;
                int tokenIndex = tokenIndices[b * tokenCount + t];
                float* tokenEmbeddingsGradVector = δtokenEmbeddings + tokenIndex * channelCount;
                float* positionEmbeddingsGradVector = δpositionEmbeddings + t * channelCount;
                for (int c = 0; c < channelCount; c++)
                {
                    float derivative = outputGrad_bt[c];
                    tokenEmbeddingsGradVector[c] += derivative;
                    positionEmbeddingsGradVector[c] += derivative;
                }
            }
        }
    }

    /// <summary>
    /// Forward pass of Layer Normalization.
    /// </summary>
    /// <param name="input">The input tensor of shape [batchSize, tokenCount, channelCount].</param>
    /// <param name="weight">The weight tensor of shape [channelCount].</param>
    /// <param name="bias">The bias tensor of shape [channelCount].</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="tokenCount">The token count.</param>
    /// <param name="channelCount">The channel count.</param>
    /// <param name="mean">The mean tensor of shape [batchSize, tokenCount].</param>
    /// <param name="invStdDev">The inverse standard deviation tensor of shape [batchSize, tokenCount].</param>
    /// <param name="output">The output tensor of shape [batchSize, tokenCount, channelCount].</param>
    public unsafe static void LayerNormForward(
        // [batchSize, tokenCount, channelCount], [channelCount], [channelCount]
        float* input, float* weight, float* bias,
        int batchSize, int tokenCount, int channelCount,
        // [batchSize, tokenCount], [batchSize, tokenCount], [batchSize, tokenCount, channelCount]
        float* mean, float* invStdDev, float* output)
    {
        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both input and output are [batchSize,tokenCount,channelCount] of the activations
        // mean and invStdDev are [batchSize,tokenCount] buffers, to be used later in backward pass
        // at each position (b,t) of the input, the channelCount-dimensional vector
        // of activations gets normalized, then scaled and shifted
        const float eps = 1e-5f;
        P.ForRanges(batchSize, tokenCount, (b, t) =>
        {
            float* input_b = input + b * tokenCount * channelCount;
            float* mean_b = mean + b * tokenCount;
            float* invStdDev_b = invStdDev + b * tokenCount;
            float* output_b = output + b * tokenCount * channelCount;
            LayerNormForwardAtBatchToken(input_b, weight, bias,
                channelCount, t, eps,
                mean_b, invStdDev_b, output_b);
        });
        //for (int b = 0; b < batchSize; b++)
        //{
        //    float* input_b = input + b * tokenCount * channelCount;
        //    float* mean_b = mean + b * tokenCount;
        //    float* invStdDev_b = invStdDev + b * tokenCount;
        //    float* output_b = output + b * tokenCount * channelCount;
        //    for (int t = 0; t < tokenCount; t++)
        //    {
        //        LayerNormForwardAtBatchToken(input_b, weight, bias,
        //            channelCount, t, eps,
        //            mean_b, invStdDev_b, output_b);
        //    }
        //}

        static unsafe void LayerNormForwardAtBatchToken(
            float* input_b, float* weight, float* bias, int channelCount,
            int t, float eps, float* mean_b, float* invStdDev_b, float* output_b)
        {
            // seek to the input position input[b,t,:]
            float* x = input_b + t * channelCount;
            // calculate the mean
            float m = 0.0f;
            for (int c = 0; c < channelCount; c++)
            {
                m += x[c];
            }
            m /= channelCount;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < channelCount; i++)
            {
                float xMinusMean = x[i] - m;
                v += xMinusMean * xMinusMean;
            }
            v /= channelCount;
            // calculate the invStdDev (reciprocal standard deviation)
            float s = 1.0f / MathF.Sqrt(v + eps);
            // seek to the output position in output[b,t,:]
            float* output_bt = output_b + t * channelCount;
            for (int c = 0; c < channelCount; c++)
            {
                float n = (s * (x[c] - m)); // normalize
                float o = n * weight[c] + bias[c]; // scale and shift
                output_bt[c] = o; // write
            }
            // cache the mean and invStdDev for the backward pass later
            mean_b[t] = m;
            invStdDev_b[t] = s;
        }
    }

    /// <summary>
    /// Backward pass of Layer Normalization.
    /// </summary>
    /// <param name="δoutput">The gradients of the output tensor. Shape: [batchSize, tokenCount, channelCount].</param>
    /// <param name="input">The input tensor. Shape: [batchSize, tokenCount, channelCount].</param>
    /// <param name="weight">The weight tensor. Shape: [channelCount].</param>
    /// <param name="mean">The mean tensor. Shape: [batchSize, tokenCount].</param>
    /// <param name="invStdDev">The inverse standard deviation tensor. Shape: [batchSize, tokenCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="channelCount">The number of channels.</param>
    /// <param name="δweight">The gradients of the weight tensor. Shape: [channelCount].</param>
    /// <param name="δbias">The gradients of the bias tensor. Shape: [channelCount].</param>
    /// <param name="δinput">The gradients of the input tensor. Shape: [batchSize, tokenCount, channelCount].</param>
    public unsafe static void LayerNormBackward(
        // [batchSize, tokenCount, channelCount], [batchSize, tokenCount, channelCount], [channelCount]
        float* δoutput, float* input, float* weight,
        // [batchSize, tokenCount], [batchSize, tokenCount]
        float* mean, float* invStdDev,
        int batchSize, int tokenCount, int channelCount,
        // [channelCount], [channelCount], [batchSize, tokenCount, channelCount]
        float* δweight, float* δbias, float* δinput)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                LayerNormBackwardAtBatchToken(
                    δoutput, input, weight, mean, invStdDev,
                    tokenCount, channelCount,
                    δweight, δbias, δinput,
                    b, t);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void LayerNormBackwardAtBatchToken(
            float* δoutput, float* input, float* weight, float* mean, float* invStdDev,
            int tokenCount, int channelCount, float* δweight, float* δbias,
            float* δinput, int b, int t)
        {
            float* δoutput_bt = δoutput + b * tokenCount * channelCount + t * channelCount;
            float* input_bt = input + b * tokenCount * channelCount + t * channelCount;
            float mean_bt = mean[b * tokenCount + t];
            float invStdDev_bt = invStdDev[b * tokenCount + t];

            float* δinput_bt = δinput + b * tokenCount * channelCount + t * channelCount;

            // first: two reduce operations
            float δnormMean = 0.0f;
            float δnormNormMean = 0.0f;
            for (int c = 0; c < channelCount; c++)
            {
                float δnorm_c = weight[c] * δoutput_bt[c];
                δnormMean += δnorm_c;
                float norm_btc = (input_bt[c] - mean_bt) * invStdDev_bt;
                δnormNormMean += δnorm_c * norm_btc;
            }
            δnormMean /= channelCount;
            δnormNormMean /= channelCount;

            // now iterate again and accumulate all the gradients
            for (int c = 0; c < channelCount; c++)
            {
                float norm_btc = (input_bt[c] - mean_bt) * invStdDev_bt;
                float δnorm_c = weight[c] * δoutput_bt[c];
                // gradient contribution to bias
                δbias[c] += δoutput_bt[c];
                // gradient contribution to weight
                δweight[c] += norm_btc * δoutput_bt[c];
                // gradient contribution to input (term (1, 2, 3) * final scale
                float δ = (δnorm_c - δnormMean - norm_btc * δnormNormMean) * invStdDev_bt;
                δinput_bt[c] += δ;
            }
        }
    }

    /// <summary>
    /// Matrix multiplication between the input tensor and transposed weight tensor 
    /// and optionally adds bias if not null.
    /// </summary>
    /// <remarks>
    /// Note how this is more of a vector * matrix operation than a matrix * matrix operation, 
    /// since for each batch and token it multiples input row vector with 
    /// weight (transposed) matrix and adds bias vector.
    /// </remarks>
    /// <param name="input">The input tensor of shape [batchSize, tokenCount, inputChannelCount].</param>
    /// <param name="weight">The weight tensor (transposed) of shape [outputChannelCount, inputChannelCount].</param>
    /// <param name="bias">The bias tensor of shape [outputChannelCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="inputChannelCount">The number of input channels.</param>
    /// <param name="outputChannelCount">The number of output channels.</param>
    /// <param name="output">The output tensor of shape [batchSize, tokenCount, outputChannelCount].</param>
    public unsafe static void MatMulForward(
        // [batchSize, tokenCount, inputChannelCount], [outputChannelCount, inputChannelCount], [outputChannelCount]
        float* input, float* weight, float* bias,
        int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
        // [batchSize, tokenCount, outputChannelCount]
        float* output)
    {
        // Note that weight is transposed which is great since column vector is then a row vector
        // output[B,T,:] = input [B,T,:] * weightT[:,:] + bias[:]

        //#pragma omp parallel for collapse(2)
        P.ForRanges(batchSize, tokenCount, (b, t) =>
        {
            MatMulForwardAtBatchToken(output, input, weight, bias,
                tokenCount, inputChannelCount, outputChannelCount, b, t);
        });

        // https://richardstartin.github.io/posts/mmm-revisited

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void MatMulForwardAtBatchToken(
            float* output, float* input, float* weight, float* bias,
            int tokenCount, int inputChannelCount, int outputChannelCount,
            nint b, nint t)
        {
            float* output_bt = output + b * tokenCount * outputChannelCount + t * outputChannelCount;
            float* input_bt = input + b * tokenCount * inputChannelCount + t * inputChannelCount;
            nint oc = 0;
            if (true)
            {
                const int Unroll = 4;
                for (; oc < (outputChannelCount - Unroll); oc += Unroll)
                {
                    float result0 = (bias != null) ? bias[oc + 0] : 0.0f;
                    float result1 = (bias != null) ? bias[oc + 1] : 0.0f;
                    float result2 = (bias != null) ? bias[oc + 2] : 0.0f;
                    float result3 = (bias != null) ? bias[oc + 3] : 0.0f;
                    //var results = (bias != null) ? Vector128.Load(bias + oc) : Vector128<float>.Zero;

                    float* weightRow0 = weight + (oc + 0) * inputChannelCount;
                    float* weightRow1 = weight + (oc + 1) * inputChannelCount;
                    float* weightRow2 = weight + (oc + 2) * inputChannelCount;
                    float* weightRow3 = weight + (oc + 3) * inputChannelCount;
                    var sum0 = Vector<float>.Zero;
                    var sum1 = Vector<float>.Zero;
                    var sum2 = Vector<float>.Zero;
                    var sum3 = Vector<float>.Zero;
                    nint ic = 0;
                    for (; ic < (inputChannelCount - Vector<float>.Count); ic += Vector<float>.Count)
                    {
                        //var input_btic = Vector.Load(input_bt + ic).AsVector256();
                        //sum0 = Fma.MultiplyAdd(input_btic, Vector256.Load(weightRow0 + ic), sum0.AsVector256()).AsVector();
                        //sum1 = Fma.MultiplyAdd(input_btic, Vector256.Load(weightRow1 + ic), sum1.AsVector256()).AsVector();
                        //sum2 = Fma.MultiplyAdd(input_btic, Vector256.Load(weightRow2 + ic), sum2.AsVector256()).AsVector();
                        //sum3 = Fma.MultiplyAdd(input_btic, Vector256.Load(weightRow3 + ic), sum3.AsVector256()).AsVector();
                        var input_btic = Vector.Load(input_bt + ic);
                        sum0 += input_btic * Vector.Load(weightRow0 + ic);
                        sum1 += input_btic * Vector.Load(weightRow1 + ic);
                        sum2 += input_btic * Vector.Load(weightRow2 + ic);
                        sum3 += input_btic * Vector.Load(weightRow3 + ic);
                    }

                    result0 += Vector.Sum(sum0);
                    result1 += Vector.Sum(sum1);
                    result2 += Vector.Sum(sum2);
                    result3 += Vector.Sum(sum3);
                    //var sums = Vector128.Create(Vector.Sum(sum0), Vector.Sum(sum1), Vector.Sum(sum2), Vector.Sum(sum3));
                    //results += sums;

                    for (; ic < inputChannelCount; ic++)
                    {
                        var input_btic = input_bt[ic];

                        //result0 = Single.FusedMultiplyAdd(input_btic, weightRow0[ic], result0);
                        //result1 = Single.FusedMultiplyAdd(input_btic, weightRow1[ic], result1);
                        //result2 = Single.FusedMultiplyAdd(input_btic, weightRow2[ic], result2);
                        //result3 = Single.FusedMultiplyAdd(input_btic, weightRow3[ic], result3);
                        result0 += input_btic * weightRow0[ic];
                        result1 += input_btic * weightRow1[ic];
                        result2 += input_btic * weightRow2[ic];
                        result3 += input_btic * weightRow3[ic];
                        //results += (Vector128.Create(weightRow0[ic], weightRow1[ic], weightRow2[ic], weightRow3[ic]) * input_btic);
                    }
                    output_bt[oc + 0] = result0;
                    output_bt[oc + 1] = result1;
                    output_bt[oc + 2] = result2;
                    output_bt[oc + 3] = result3;
                    //Vector128.Store(results, output_bt + oc);
                }
            }
            for (; oc < outputChannelCount; oc++)
            {
                float result = (bias != null) ? bias[oc] : 0.0f;
                float* weightRow = weight + oc * inputChannelCount;
                var sum = Vector<float>.Zero;
                nint i = 0;

                //var sum2 = Vector<float>.Zero;
                //for (; i < (inputChannelCount - Vector<float>.Count * 2); i += Vector<float>.Count * 2)
                //{
                //    sum += Vector.Load(input_bt + i) * Vector.Load(weightRow + i);
                //    sum2 += Vector.Load(input_bt + i + Vector<float>.Count) * Vector.Load(weightRow + i + Vector<float>.Count);
                //}
                //sum += sum2;

                for (; i < (inputChannelCount - Vector<float>.Count); i += Vector<float>.Count)
                {
                    sum += Vector.Load(input_bt + i) * Vector.Load(weightRow + i);
                }
                result += Vector.Sum(sum);
                for (; i < inputChannelCount; i++)
                {
                    result += input_bt[i] * weightRow[i];
                }
                output_bt[oc] = result;
            }
        }
    }

    /// <summary>
    /// Backward pass of matrix multiplication operation.
    /// </summary>
    /// <param name="δoutput">The gradient of the output tensor. Shape: [batchSize, tokenCount, outputChannelCount].</param>
    /// <param name="input">The input tensor. Shape: [batchSize, tokenCount, inputChannelCount].</param>
    /// <param name="weight">The weight tensor. Shape: [outputChannelCount, inputChannelCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="inputChannelCount">The number of input channels.</param>
    /// <param name="outputChannelCount">The number of output channels.</param>
    /// <param name="δweight">The gradient of the weight tensor. Shape: [outputChannelCount, inputChannelCount].</param>
    /// <param name="δbias">The gradient of the bias tensor. Shape: [outputChannelCount].</param>
    /// <param name="δinput">The gradient of the input tensor. Shape: [batchSize, tokenCount, inputChannelCount].</param>
    public unsafe static void MatMulBackward(
        // [batchSize, tokenCount, outputChannelCount], [batchSize, tokenCount, inputChannelCount], [outputChannelCount, inputChannelCount]
        float* δoutput, float* input, float* weight,
        int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
        // [outputChannelCount, inputChannelCount], [outputChannelCount], [batchSize, tokenCount, inputChannelCount]
        float* δweight, float* δbias, float* δinput)
    {
        // backward could be done in a single "round" of loops but that doesn't
        // afford an efficient parallelization strategy

        // backward into input first, parallelize over [batchSize,tokenCount]
        //#pragma omp parallel for collapse(2)
        P.ForRanges(batchSize, tokenCount, (b, t) =>
        {
            MatMulBackwardForInputAtBatchToken(δoutput, weight,
                tokenCount, inputChannelCount, outputChannelCount,
                δinput,
                b, t);
        });
        //}

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void MatMulBackwardForInputAtBatchToken(
            float* δoutput, float* weight,
            nint tokenCount, nint inputChannelCount, nint outputChannelCount,
            float* δinput,
            nint b, nint t)
        {
            float* δoutput_bt = δoutput + b * tokenCount * outputChannelCount + t * outputChannelCount;
            float* δinput_bt = δinput + b * tokenCount * inputChannelCount + t * inputChannelCount;
            for (nint oc = 0; oc < outputChannelCount; oc++)
            {
                float* weightRow = weight + oc * inputChannelCount;
                float δoutput_btoc = δoutput_bt[oc];
                var δoutputVector_btoc = new Vector<float>(δoutput_btoc);
                nint ic = 0;
                for (; ic < (inputChannelCount - Vector<float>.Count); ic += Vector<float>.Count)
                {
                    var δinput_btic = δinput_bt + ic;
                    var δinputVector_btic = Vector.Load(δinput_btic);
                    δinputVector_btic += Vector.Load(weightRow + ic) * δoutputVector_btoc;
                    Vector.Store(δinputVector_btic, δinput_btic);
                }
                for (; ic < inputChannelCount; ic++)
                {
                    δinput_bt[ic] += weightRow[ic] * δoutput_btoc;
                }
            }
        }

        // backward into weight/bias, parallelize over output channels
        //#pragma omp parallel for
        P.For(0, outputChannelCount, oc =>
        {
            MatMulBackwardForWeightBiasAtOutputChannel(
                δoutput, input,
                batchSize, tokenCount, inputChannelCount, outputChannelCount,
                δweight, δbias,
                oc);
        });
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void MatMulBackwardForWeightBiasAtOutputChannel(
            float* δoutput, float* input,
            nint batchSize, nint tokenCount, nint inputChannelCount, nint outputChannelCount,
            float* δweight, float* δbias,
            nint oc)
        {
            for (nint b = 0; b < batchSize; b++)
            {
                for (nint t = 0; t < tokenCount; t++)
                {
                    float* δoutput_bt = δoutput + b * tokenCount * outputChannelCount + t * outputChannelCount;
                    float* input_bt = input + b * tokenCount * inputChannelCount + t * inputChannelCount;
                    float* δweightRow = δweight + oc * inputChannelCount;
                    float δoutput_btoc = δoutput_bt[oc];
                    if (δbias != null) { δbias[oc] += δoutput_btoc; }
                    nint ic = 0;
                    var δoutputVector_btoc = new Vector<float>(δoutput_btoc);
                    for (; ic < (inputChannelCount - Vector<float>.Count); ic += Vector<float>.Count)
                    {
                        var δweightPtr = δweightRow + ic;
                        var δweightVector = Vector.Load(δweightPtr);
                        δweightVector += Vector.Load(input_bt + ic) * δoutputVector_btoc;
                        Vector.Store(δweightVector, δweightPtr);
                    }
                    for (; ic < inputChannelCount; ic++)
                    {
                        δweightRow[ic] += input_bt[ic] * δoutput_btoc;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Forward pass of the Attention layer.
    /// </summary>
    /// <param name="input">The input tensor of shape [batchSize, tokenCount, 3 * channelCount] holding the query, key, and value (Q, K, V) vectors.</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens in the sequence.</param>
    /// <param name="channelCount">The number of channels in the input tensor.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="preAttention">The pre-attention tensor of shape [batchSize, headCount, tokenCount, tokenCount] that holds the pre-attention scores.</param>
    /// <param name="postAttention">The post-attention tensor of shape [batchSize, headCount, tokenCount, tokenCount] that holds the post-attention scores.</param>
    /// <param name="output">The output tensor of shape [batchSize, tokenCount, channelCount] that holds the attention output.</param>
    public unsafe static void AttentionForward(
        // [batchSize, tokenCount, 3 * channelCount (query Q, key K, value V)]
        float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        // [batchSize, headCount, tokenCount, tokenCount], [batchSize, headCount, tokenCount, tokenCount], [batchSize, tokenCount, channelCount]
        float* preAttention, float* postAttention, float* output)
    {
        // input is (batchSize, tokenCount, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (batchSize, headCount, tokenCount, tokenCount).
        // headCount = number of heads, tokenCount = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (batchSize, tokenCount, channelCount)
        // attention is the only layer that mixes information across time/token sequence
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int qkvChannelCount = channelCount * 3;
        int headSize = channelCount / headCount; // head size
        Debug.Assert(channelCount == (headSize * headCount));
        float scale = 1.0f / MathF.Sqrt(headSize);

        // Scaled Dot-Product Attention as ASCII art:
        //
        //          MatMul
        //          ↑    ↑
        //    Softmax    ↑
        //      ↑        ↑
        //    Mask       ↑
        //      ↑        ↑
        //    Scale      ↑
        //      ↑        ↑
        //    MatMul     ↑
        //    ↑    ↑     ↑
        //    Q    K     V
        //
        // Code below works on each individual batch sample, token, and head in parallel

        //#pragma omp parallel for collapse(3)
        P.ForRanges(batchSize, tokenCount, headCount, (b, t, h) =>
        {
            AttentionForwardAtBatchTokenHead(input,
                tokenCount, channelCount, headCount,
                preAttention, postAttention, output,
                b, t, h);
        });

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void AttentionForwardAtBatchTokenHead(float* input,
            int tokenCount, int channelCount, int headCount,
            float* preAttention, float* postAttention, float* output,
            int b, int t, int h)
        {
            int qkvChannelCount = channelCount * 3;
            int headSize = channelCount / headCount;
            Debug.Assert(channelCount == (headSize * headCount));
            float scale = 1.0f / MathF.Sqrt(headSize);

            float* query_t = input + b * tokenCount * qkvChannelCount + t * qkvChannelCount + h * headSize;
            float* preAtt_bth = preAttention + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
            float* postAtt_bth = postAttention + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;

            // pass 1: calculate query dot key and max (for softmax)
            var key_bh = input + b * tokenCount * qkvChannelCount + h * headSize + channelCount * 1; // +channelCount because it's 
            float max = float.MinValue;
            for (int t2 = 0; t2 <= t; t2++) // note: includes t == tokenCount
            {
                float* key_t2 = key_bh + t2 * qkvChannelCount;

                // (query_t) dot (key_t2)
                float dot = 0.0f;
                for (int i = 0; i < headSize; i++)
                {
                    dot += query_t[i] * key_t2[i];
                }
                dot *= scale;
                max = MathF.Max(max, dot);
                preAtt_bth[t2] = dot;
            }

            // pass 2: calculate the exp and keep track of sum
            // max is being calculated and subtracted only for numerical stability
            float expSum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++)
            {
                float exp = MathF.Exp(preAtt_bth[t2] - max);
                expSum += exp;
                postAtt_bth[t2] = exp;
            }
            float invExpSum = expSum == 0.0f ? 0.0f : 1.0f / expSum;

            // pass 3: normalize to get the softmax
            for (int t2 = 0; t2 < tokenCount; t2++)
            {
                if (t2 <= t)
                {
                    postAtt_bth[t2] *= invExpSum;
                }
                else
                {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    postAtt_bth[t2] = 0.0f;
                }
            }

            // pass 4: accumulate weighted values into the output of attention
            float* output_bth = output + b * tokenCount * channelCount + t * channelCount + h * headSize;
            for (int i = 0; i < headSize; i++) { output_bth[i] = 0.0f; }

            float* value_bh = input + b * tokenCount * qkvChannelCount + h * headSize + channelCount * 2; // +channelCount*2 because it's value
            for (int t2 = 0; t2 <= t; t2++)
            {
                float* value_t2 = value_bh + t2 * qkvChannelCount;
                float att_btht2 = postAtt_bth[t2];
                for (int i = 0; i < headSize; i++)
                {
                    output_bth[i] += att_btht2 * value_t2[i];
                }
            }
        }
    }

    /// <summary>
    /// Backward pass of the Attention layer.
    /// </summary>
    /// <param name="δoutput">The gradient of the output tensor. Shape: [batchSize, tokenCount, channelCount].</param>
    /// <param name="postAttention">The post-softmax attention tensor. Shape: [batchSize, headCount, tokenCount, tokenCount].</param>
    /// <param name="input">The input tensor. Shape: [batchSize, tokenCount, 3 * channelCount (Q, K, V)].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="channelCount">The number of channels.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="δpreAttention">The gradient of the pre-softmax attention tensor. Shape: [batchSize, headCount, tokenCount, tokenCount].</param>
    /// <param name="δpostAttention">The gradient of the attention tensor. Shape: [batchSize, headCount, tokenCount, tokenCount].</param>
    /// <param name="δinput">The gradient of the input tensor. Shape: [batchSize, tokenCount, 3 * channelCount (Q, K, V)].</param>
    public unsafe static void AttentionBackward(
        // [batchSize, tokenCount, channelCount], [batchSize, headCount, tokenCount, tokenCount], [batchSize, tokenCount, 3 * channelCount (Q, K, V)]
        float* δoutput, float* postAttention, float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        // [batchSize, headCount, tokenCount, tokenCount], [batchSize, headCount, tokenCount, tokenCount], [batchSize, tokenCount, 3 * channelCount (Q, K, V)]
        float* δpreAttention, float* δpostAttention, float* δinput)
    {
        int qkvChannelCount = channelCount * 3;
        int headSize = channelCount / headCount;
        float scale = 1.0f / MathF.Sqrt(headSize);

        //for (int b = 0; b < batchSize; b++)
        //{
        //    for (int t = 0; t < tokenCount; t++)
        //    {
        //        for (int h = 0; h < headCount; h++)
        //        {
        //            AttentionBackwardAtBatchTokenHead(δoutput, postAttention, input,
        //                tokenCount, channelCount, headCount, qkvChannelCount, headSize, scale,
        //                δpreAttention, δpostAttention, δinput, b, t, h);
        //        }
        //    }
        //}
        // Cannot simply parallize like this since some derivatives are "shared"
        // like δinput (derivative of input which is output from this method)
        //P.ForRanges(batchSize, tokenCount, headCount, (b, t, h) =>
        //{
        //    AttentionBackwardAtBatchTokenHead(δoutput, postAttention, input,
        //        tokenCount, channelCount, headCount, qkvChannelCount, headSize, scale,
        //        δpreAttention, δpostAttention, δinput, b, t, h);
        //});
        P.ForRanges(batchSize, headCount, (b, h) =>
        {
            for (int t = 0; t < tokenCount; t++)
            {
                AttentionBackwardAtBatchTokenHead(δoutput, postAttention, input, tokenCount, channelCount, headCount, qkvChannelCount, headSize, scale, δpreAttention, δpostAttention, δinput, b, t, h);
            }
        });

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void AttentionBackwardAtBatchTokenHead(
            float* δoutput, float* postAttention, float* input,
            int tokenCount, int channelCount, int headCount,
            int qkvChannelCount, int headSize, float scale,
            float* δpreAttention, float* δpostAttention, float* δinput,
            int b, int t, int h)
        {
            float* att_bth = postAttention + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
            float* datt_bth = δpostAttention + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
            float* dpreatt_bth = δpreAttention + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
            float* dquery_t = δinput + b * tokenCount * qkvChannelCount + t * qkvChannelCount + h * headSize;
            float* query_t = input + b * tokenCount * qkvChannelCount + t * qkvChannelCount + h * headSize;

            // backward pass 4, through the value accumulation
            float* δoutput_bth = δoutput + b * tokenCount * channelCount + t * channelCount + h * headSize;
            for (int t2 = 0; t2 <= t; t2++)
            {
                float* value_t2 = input + b * tokenCount * qkvChannelCount + t2 * qkvChannelCount + h * headSize + channelCount * 2; // +channelCount*2 because it's value
                float* dvalue_t2 = δinput + b * tokenCount * qkvChannelCount + t2 * qkvChannelCount + h * headSize + channelCount * 2;
                int i = 0;
                var att_bth_t2 = att_bth[t2];
                var datt_bth_sum = 0f;
                if (Vector<float>.IsSupported)
                {
                    var att_bth_broadcastVector = new Vector<float>(att_bth_t2);
                    var datt_bth_sum_vector = Vector<float>.Zero;
                    for (; i < headSize - Vector<float>.Count; i += Vector<float>.Count)
                    {
                        var valueVector = Vector.Load(value_t2 + i);
                        var doutputVector = Vector.Load(δoutput_bth + i);
                        datt_bth_sum_vector += valueVector * doutputVector;
                        var dValuePtr = dvalue_t2 + i;
                        var dValueVector = Vector.Load(dValuePtr);
                        dValueVector += att_bth_broadcastVector * doutputVector;
                        Vector.Store(dValueVector, dValuePtr);
                    }
                    datt_bth_sum = Vector.Sum(datt_bth_sum_vector);
                }
                for (; i < headSize; i++)
                {
                    // in the forward pass this was:
                    // output_bth[i] += att_bth[t2] * value_t2[i];
                    // so now we have:
                    //datt_bth[t2] += value_t2[i] * δoutput_bth[i];
                    //dvalue_t2[i] += att_bth[t2] * δoutput_bth[i];
                    datt_bth_sum += value_t2[i] * δoutput_bth[i];
                    dvalue_t2[i] += att_bth_t2 * δoutput_bth[i];
                }
                datt_bth[t2] += datt_bth_sum;
            }

            // backward pass 2 & 3, the softmax
            // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
            for (int t2 = 0; t2 <= t; t2++)
            {
                var att_bth_t2 = att_bth[t2];
                var datt_bth_t2 = datt_bth[t2];
                for (int t3 = 0; t3 <= t; t3++)
                {
                    float indicator = t2 == t3 ? 1.0f : 0.0f;
                    float local_derivative = att_bth_t2 * (indicator - att_bth[t3]);
                    dpreatt_bth[t3] += local_derivative * datt_bth_t2;
                }
            }

            // backward pass 1, the query @ key matmul
            for (int t2 = 0; t2 <= t; t2++)
            {
                float* key_t2 = input + b * tokenCount * qkvChannelCount + t2 * qkvChannelCount + h * headSize + channelCount; // +channelCount because it's key
                float* dkey_t2 = δinput + b * tokenCount * qkvChannelCount + t2 * qkvChannelCount + h * headSize + channelCount; // +channelCount because it's key
                var dpreatt_bth_t2_scaled = dpreatt_bth[t2] * scale;
                for (int i = 0; i < headSize; i++)
                {
                    // in the forward pass this was:
                    // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                    // so now we have:
                    dquery_t[i] += key_t2[i] * dpreatt_bth_t2_scaled;
                    dkey_t2[i] += query_t[i] * dpreatt_bth_t2_scaled;
                }
            }
        }
    }

    static readonly float GeluScalingFactor = MathF.Sqrt(2.0f / MathF.PI);
    public unsafe static void GeLUForward(float* input, int count, float* output)
    {
        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        // TODO: Chunk!
        P.For(0, count, i =>
        {
            float x = input[i];
            float cube = 0.044715f * x * x * x;
            output[i] = 0.5f * x * (1.0f + MathF.Tanh(GeluScalingFactor * (x + cube)));
        });
        //for (int i = 0; i < count; i++)
        //{
        //    float x = input[i];
        //    float cube = 0.044715f * x * x * x;
        //    output[i] = 0.5f * x * (1.0f + MathF.Tanh(GELU_SCALING_FACTOR * (x + cube)));
        //}
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void GeLUBackward(
        float* δoutput, float* input,
        int count, float* δinput)
    {
        // TODO: Chunk!
        P.For(0, count, i =>
        {
            float x = input[i];
            var grad = GeLUBackward(x);
            δinput[i] += grad * δoutput[i];
        });
        //for (int i = 0; i < count; i++)
        //{
        //    float x = input[i];
        //    var grad = GeLUBackward(x);
        //    δinput[i] += grad * δoutput[i];
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static float GeLUBackward(float x)
        {
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GeluScalingFactor * (x + cube);
            float tanh_out = MathF.Tanh(tanh_arg);
            float coshf_out = MathF.Cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float grad = 0.5f * (1.0f + tanh_out) +
                x * 0.5f * sech_out * GeluScalingFactor * (1.0f + 3.0f * 0.044715f * x * x);
            return grad;
        }
    }

    public unsafe static void ResidualForward(
        float* left, float* right, int count, float* output)
    {
        for (int i = 0; i < count; i++)
        {
            output[i] = left[i] + right[i];
        }
    }

    public unsafe static void ResidualBackward(
        float* δoutput, int count, float* δleft, float* δright)
    {
        for (int i = 0; i < count; i++)
        {
            δleft[i] += δoutput[i];
            δright[i] += δoutput[i];
        }
    }

    public unsafe static void SoftmaxForward(
        float* logits,
        int batchSize, int tokenCount, int vocabularySize,
        float* probabilities)
    {
        // output: probabilities are (batchSize,tokenCount,vocabularySize) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (batchSize,tokenCount,vocabularySize) of the unnormalized log probabilities
        //#pragma omp parallel for collapse(2)
        //for (b = 0; b < batchSize; b++)
        P.ForRanges(batchSize, tokenCount, (b, t) =>
        {
            // probabilities <- softmax(logits)
            float* logits_bt = logits + b * tokenCount * vocabularySize + t * vocabularySize;
            float* probs_bt = probabilities + b * tokenCount * vocabularySize + t * vocabularySize;

            // max is only calculated and subtracted for numerical stability
            float max = float.MinValue;
            for (int i = 0; i < vocabularySize; i++)
            {
                max = MathF.Max(max, logits_bt[i]);
            }
            float sum = 0.0f;
            for (int i = 0; i < vocabularySize; i++)
            {
                probs_bt[i] = MathF.Exp(logits_bt[i] - max);
                sum += probs_bt[i];
            }
            var invSum = 1.0f / sum;
            for (int i = 0; i < vocabularySize; i++)
            {
                probs_bt[i] *= invSum;
            }
        });
    }

    public unsafe static void CrossEntropyForward(
        float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* losses)
    {
        // output: losses is (batchSize,tokenCount) of the individual losses at each position
        // input: probabilities are (batchSize,tokenCount,vocabularySize) of the probabilities
        // input: targetTokenIndices is (batchSize,tokenCount) of integers giving the correct index in logits
        for (int b = 0; b < batchSize; b++)
        {
            float* probs_b = probabilities + b * tokenCount * vocabularySize;
            var tokenIndices_b = targetTokenIndices + b * tokenCount;
            for (int t = 0; t < tokenCount; t++)
            {
                // loss = -log(probabilities[target])
                float* probs_bt = probs_b + t * vocabularySize;
                int tokenIndex = tokenIndices_b[t];
                losses[b * tokenCount + t] = -MathF.Log(probs_bt[tokenIndex]);
            }
        }
    }

    public unsafe static void CrossEntropySoftmaxBackward(
        float* δlosses, float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* δlogits)
    {
        // backwards through both softmax and crossentropy
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                float* dlogits_bt = δlogits + b * tokenCount * vocabularySize + t * vocabularySize;
                float* probs_bt = probabilities + b * tokenCount * vocabularySize + t * vocabularySize;
                float dloss = δlosses[b * tokenCount + t];
                int tokenIndex = targetTokenIndices[b * tokenCount + t];
                for (int i = 0; i < vocabularySize; i++)
                {
                    float p = probs_bt[i];
                    float indicator = i == tokenIndex ? 1.0f : 0.0f;
                    dlogits_bt[i] += (p - indicator) * dloss;
                }
            }
        }
    }
}
