﻿using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using P = nietras.LargeLanguageModel.LlmParallel;
//using P = nietras.LargeLanguageModel.NotParallel;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

// all the individual layers' forward and backward passes
// batchSize = B, tokenCount = T, channelCount = C, vocabularySize = V
public static partial class Llm
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
    /// Forward pass of LayerNorm layer.
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
        for (int b = 0; b < batchSize; b++)
        {
            float* mean_b = mean + b * tokenCount;
            float* invStdDev_b = invStdDev + b * tokenCount;
            float* output_b = output + b * tokenCount * channelCount;
            for (int t = 0; t < tokenCount; t++)
            {
                // seek to the input position input[b,t,:]
                float* x = input + b * tokenCount * channelCount + t * channelCount;
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
    }

    public unsafe static void LayerNormBackward(
        float* δoutput, float* input, float* weight, float* mean, float* invStdDev,
        int batchSize, int tokenCount, int channelCount,
        float* δweight, float* δbias, float* δinput)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                LayerNormBackwardAtBatchToken(
                    δoutput, input, weight, mean, invStdDev,
                    tokenCount, channelCount, b, t,
                    δweight, δbias, δinput);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void LayerNormBackwardAtBatchToken(
            float* δoutput, float* input, float* weight, float* mean, float* invStdDev,
            int tokenCount, int channelCount, int b, int t,
            float* δweight, float* δbias, float* δinput)
        {
            float* δoutput_bt = δoutput + b * tokenCount * channelCount + t * channelCount;
            float* input_bt = input + b * tokenCount * channelCount + t * channelCount;
            float mean_bt = mean[b * tokenCount + t];
            float invStdDev_bt = invStdDev[b * tokenCount + t];

            float* δinput_bt = δinput + b * tokenCount * channelCount + t * channelCount;

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int c = 0; c < channelCount; c++)
            {
                float norm_btc = (input_bt[c] - mean_bt) * invStdDev_bt;
                float dnorm_c = weight[c] * δoutput_bt[c];
                dnorm_mean += dnorm_c;
                dnorm_norm_mean += dnorm_c * norm_btc;
            }
            dnorm_mean /= channelCount;
            dnorm_norm_mean /= channelCount;

            // now iterate again and accumulate all the gradients
            for (int c = 0; c < channelCount; c++)
            {
                float norm_btc = (input_bt[c] - mean_bt) * invStdDev_bt;
                float dnorm_c = weight[c] * δoutput_bt[c];
                // gradient contribution to bias
                δbias[c] += δoutput_bt[c];
                // gradient contribution to weight
                δweight[c] += norm_btc * δoutput_bt[c];
                // gradient contribution to input
                float δ = (dnorm_c - dnorm_mean - norm_btc * dnorm_norm_mean)
                    * invStdDev_bt; // final scale
                δinput_bt[c] += δ;
            }
        }
    }

    public unsafe static void MatMulForward(float* output,
                        float* input, float* weight, float* bias,
                        int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount)
    {
        // most of the running time is spent here and in matmul_backward
        // OC is short for "output channels"
        // input is (batchSize,tokenCount,channelCount), weight is (OC, channelCount), bias is (OC)
        // output will be (batchSize,tokenCount,OC)
        //#pragma omp parallel for collapse(2)
        P.ForRanges(batchSize, tokenCount, (b, t) =>
        {
            MatMulForwardAtBatchToken(output, input, weight, bias, tokenCount, inputChannelCount, outputChannelCount, b, t);
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
            for (nint o = 0; o < outputChannelCount; o++)
            {
                float result = (bias != null) ? bias[o] : 0.0f;
                float* wrow = weight + o * inputChannelCount;
                var sum = Vector<float>.Zero;
                nint i = 0;

                //var sum2 = Vector<float>.Zero;
                //for (; i < (inputChannelCount - Vector<float>.Count * 2); i += Vector<float>.Count * 2)
                //{
                //    sum += Vector.Load(input_bt + i) * Vector.Load(wrow + i);
                //    sum2 += Vector.Load(input_bt + i + Vector<float>.Count) * Vector.Load(wrow + i + Vector<float>.Count);
                //}
                //sum += sum2;

                for (; i < (inputChannelCount - Vector<float>.Count); i += Vector<float>.Count)
                {
                    sum += Vector.Load(input_bt + i) * Vector.Load(wrow + i);
                }
                result += Vector.Sum(sum);
                for (; i < inputChannelCount; i++)
                {
                    result += input_bt[i] * wrow[i];
                }
                output_bt[o] = result;
            }
        }
    }

    public unsafe static void MatMulBackward(float* δinput, float* δweight, float* δbias,
                         float* δoutput, float* input, float* weight,
                         int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount)
    {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy

        // backward into input first, parallelize over batchSize,tokenCount
        //#pragma omp parallel for collapse(2)
        P.ForRanges(batchSize, tokenCount, (b, t) =>
        {
            MatMulBackwardForInputAtBatchToken(δinput, δoutput, weight, tokenCount, inputChannelCount, outputChannelCount, b, t);
        });
        //}
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void MatMulBackwardForInputAtBatchToken(
            float* δinput, float* δoutput, float* weight,
            nint tokenCount, nint inputChannelCount, nint outputChannelCount,
            nint b, nint t)
        {
            float* δoutput_bt = δoutput + b * tokenCount * outputChannelCount + t * outputChannelCount;
            float* δinput_bt = δinput + b * tokenCount * inputChannelCount + t * inputChannelCount;
            for (nint o = 0; o < outputChannelCount; o++)
            {
                float* wrow = weight + o * inputChannelCount;
                float d = δoutput_bt[o];
                var dVec = new Vector<float>(d);
                nint i = 0;
                for (; i < (inputChannelCount - Vector<float>.Count); i += Vector<float>.Count)
                {
                    var δinput_bt_start = δinput_bt + i;
                    var δinput_bt_Vec = Vector.Load(δinput_bt_start);
                    δinput_bt_Vec += Vector.Load(wrow + i) * dVec;
                    Vector.Store(δinput_bt_Vec, δinput_bt_start);
                }
                for (; i < inputChannelCount; i++)
                {
                    δinput_bt[i] += wrow[i] * d;
                }
            }
        }

        // backward into weight/bias, parallelize over output channels OC
        //#pragma omp parallel for
        P.For(0, outputChannelCount, o =>
        {
            MatMulBackwardParametersAtOutputChannel(δweight, δbias, δoutput, input, batchSize, tokenCount, inputChannelCount, outputChannelCount, o);
        });
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void MatMulBackwardParametersAtOutputChannel(
            float* δweight, float* δbias, float* δoutput, float* input,
            nint batchSize, nint tokenCount, nint inputChannelCount, nint outputChannelCount,
            nint o)
        {
            for (nint b = 0; b < batchSize; b++)
            {
                for (nint t = 0; t < tokenCount; t++)
                {
                    float* δoutput_bt = δoutput + b * tokenCount * outputChannelCount + t * outputChannelCount;
                    float* input_bt = input + b * tokenCount * inputChannelCount + t * inputChannelCount;
                    float* dwrow = δweight + o * inputChannelCount;
                    float d = δoutput_bt[o];
                    if (δbias != null) { δbias[o] += d; }
                    nint i = 0;
                    var dVec = new Vector<float>(d);
                    for (; i < (inputChannelCount - Vector<float>.Count); i += Vector<float>.Count)
                    {
                        var dwstart = dwrow + i;
                        var dwVec = Vector.Load(dwstart);
                        dwVec += Vector.Load(input_bt + i) * dVec;
                        Vector.Store(dwVec, dwstart);
                    }
                    for (; i < inputChannelCount; i++)
                    {
                        dwrow[i] += input_bt[i] * d;
                    }
                }
            }
        }
    }

    public unsafe static void AttentionForward(float* output, float* preatt, float* att,
                           float* input,
                           int batchSize, int tokenCount, int channelCount, int headCount)
    {
        // input is (batchSize, tokenCount, 3C) holding the query, key, value (Q, K, vocabularySize) vectors
        // preatt, att are (batchSize, headCount, tokenCount, tokenCount). headCount = number of heads, tokenCount = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (batchSize, tokenCount, channelCount)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int C3 = channelCount * 3;
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
            AttentionForwardAtBatchTokenHead(output, preatt, att, input, tokenCount, channelCount, headCount, b, t, h, C3, headSize, scale);
        });

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void AttentionForwardAtBatchTokenHead(float* output, float* preatt, float* att, float* input,
            int tokenCount, int channelCount, int headCount, int b, int t, int h, int C3, int headSize, float scale)
        {
            float* query_t = input + b * tokenCount * C3 + t * C3 + h * headSize;
            float* preatt_bth = preatt + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
            float* att_bth = att + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;

            // pass 1: calculate query dot key and maxval
            float maxval = float.MinValue;
            for (int t2 = 0; t2 <= t; t2++) // note: includes t == tokenCount
            {
                float* key_t2 = input + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount; // +channelCount because it's key

                // (query_t) dot (key_t2)
                float val = 0.0f;
                for (int i = 0; i < headSize; i++)
                {
                    val += query_t[i] * key_t2[i];
                }
                val *= scale;
                if (val > maxval)
                {
                    maxval = val;
                }

                preatt_bth[t2] = val;
            }

            // pass 2: calculate the exp and keep track of sum
            // maxval is being calculated and subtracted only for numerical stability
            float expsum = 0.0f;
            for (int t2 = 0; t2 <= t; t2++)
            {
                float expv = MathF.Exp(preatt_bth[t2] - maxval);
                expsum += expv;
                att_bth[t2] = expv;
            }
            float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

            // pass 3: normalize to get the softmax
            for (int t2 = 0; t2 < tokenCount; t2++)
            {
                if (t2 <= t)
                {
                    att_bth[t2] *= expsum_inv;
                }
                else
                {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    att_bth[t2] = 0.0f;
                }
            }

            // pass 4: accumulate weighted values into the output of attention
            float* output_bth = output + b * tokenCount * channelCount + t * channelCount + h * headSize;
            for (int i = 0; i < headSize; i++) { output_bth[i] = 0.0f; }
            for (int t2 = 0; t2 <= t; t2++)
            {
                float* value_t2 = input + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount * 2; // +channelCount*2 because it's value
                float att_btht2 = att_bth[t2];
                for (int i = 0; i < headSize; i++)
                {
                    output_bth[i] += att_btht2 * value_t2[i];
                }
            }
        }
    }

    public unsafe static void AttentionBackward(float* δinput, float* dpreatt, float* datt,
                            float* δoutput, float* input, float* att,
                            int batchSize, int tokenCount, int channelCount, int headCount)
    {
        // input/δinput are (batchSize, tokenCount, 3C) Q,K,vocabularySize
        // att/datt/dpreatt are (batchSize, headCount, tokenCount, tokenCount)
        // δoutput is (batchSize, tokenCount, channelCount)
        int C3 = channelCount * 3;
        int headSize = channelCount / headCount; // head size
        float scale = 1.0f / MathF.Sqrt(headSize);

        //for (int b = 0; b < batchSize; b++)
        //{
        //    for (int t = 0; t < tokenCount; t++)
        //    {
        //        for (int h = 0; h < headCount; h++)
        //        {
        //            AttentionBackwardAtBatchTokenHead(δinput, dpreatt, datt, δoutput, input, att, tokenCount, channelCount, headCount, C3, headSize, scale, b, t, h);
        //        }
        //    }
        //}
        // Cannot simply parallize like this since some derivatives are "shared"
        // like δinput (derivative of input which is output from this method )
        //P.ForRanges(batchSize, tokenCount, headCount, (b, t, h) =>
        //{
        //    AttentionBackwardAtBatchTokenHead(δinput, dpreatt, datt, δoutput, input, att, tokenCount, channelCount, headCount, C3, headSize, scale, b, t, h);
        //});
        P.ForRanges(batchSize, headCount, (b, h) =>
        {
            for (int t = 0; t < tokenCount; t++)
            {
                AttentionBackwardAtBatchTokenHead(δinput, dpreatt, datt, δoutput, input, att, tokenCount, channelCount, headCount, C3, headSize, scale, b, t, h);
            }
        });

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        static unsafe void AttentionBackwardAtBatchTokenHead(float* δinput, float* dpreatt, float* datt, float* δoutput, float* input, float* att,
            int tokenCount, int channelCount, int headCount, int C3, int headSize, float scale, int b, int t, int h)
        {
            float* att_bth = att + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
            float* datt_bth = datt + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
            float* dpreatt_bth = dpreatt + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
            float* dquery_t = δinput + b * tokenCount * C3 + t * C3 + h * headSize;
            float* query_t = input + b * tokenCount * C3 + t * C3 + h * headSize;

            // backward pass 4, through the value accumulation
            float* δoutput_bth = δoutput + b * tokenCount * channelCount + t * channelCount + h * headSize;
            for (int t2 = 0; t2 <= t; t2++)
            {
                float* value_t2 = input + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount * 2; // +channelCount*2 because it's value
                float* dvalue_t2 = δinput + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount * 2;
                int i = 0;
                var att_bth_t2 = att_bth[t2];
                var datt_bth_sum = 0f;
                if (Vector<float>.IsSupported)
                {
                    var att_bth_broadcastVector = new Vector<float>(att_bth_t2);
                    var datt_bth_sum_vector = Vector<float>.Zero;
                    for (; i < headSize - Vector<float>.Count; i += Vector<float>.Count)
                    {
                        // in the forward pass this was:
                        // output_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
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
                float* key_t2 = input + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount; // +channelCount because it's key
                float* dkey_t2 = δinput + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount; // +channelCount because it's key
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

    static readonly float GELU_SCALING_FACTOR = MathF.Sqrt(2.0f / MathF.PI);
    public unsafe static void GeLUForward(float* output, float* input, int count)
    {
        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        // TODO: Chunk!
        P.For(0, count, i =>
        {
            float x = input[i];
            float cube = 0.044715f * x * x * x;
            output[i] = 0.5f * x * (1.0f + MathF.Tanh(GELU_SCALING_FACTOR * (x + cube)));
        });
        //for (int i = 0; i < count; i++)
        //{
        //    float x = input[i];
        //    float cube = 0.044715f * x * x * x;
        //    output[i] = 0.5f * x * (1.0f + MathF.Tanh(GELU_SCALING_FACTOR * (x + cube)));
        //}
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe static void GeLUBackward(float* δinput, float* input, float* δoutput, int count)
    {
        // TODO: Chunk!
        P.For(0, count, i =>
        {
            float x = input[i];
            var local_grad = δGeLU(x);
            δinput[i] += local_grad * δoutput[i];
        });
        //for (int i = 0; i < count; i++)
        //{
        //    float x = input[i];
        //    var local_grad = δGeLU(x);
        //    δinput[i] += local_grad * δoutput[i];
        //}

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static unsafe float δGeLU(float x)
        {
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = MathF.Tanh(tanh_arg);
            float coshf_out = MathF.Cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            return local_grad;
        }
    }

    public unsafe static void ResidualForward(float* output, float* input1, float* input2, int count)
    {
        for (int i = 0; i < count; i++)
        {
            output[i] = input1[i] + input2[i];
        }
    }

    public unsafe static void ResidualBackward(float* dinput1, float* dinput2, float* δoutput, int count)
    {
        for (int i = 0; i < count; i++)
        {
            dinput1[i] += δoutput[i];
            dinput2[i] += δoutput[i];
        }
    }

    public unsafe static void SoftmaxForward(float* probs, float* logits, int batchSize, int tokenCount, int vocabularySize)
    {
        // output: probs are (batchSize,tokenCount,vocabularySize) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (batchSize,tokenCount,vocabularySize) of the unnormalized log probabilities
        //#pragma omp parallel for collapse(2)
        //for (b = 0; b < batchSize; b++)
        P.ForRanges(batchSize, tokenCount, (b, t) =>
        {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * tokenCount * vocabularySize + t * vocabularySize;
            float* probs_bt = probs + b * tokenCount * vocabularySize + t * vocabularySize;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = float.MinValue;
            for (int i = 0; i < vocabularySize; i++)
            {
                if (logits_bt[i] > maxval)
                {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < vocabularySize; i++)
            {
                probs_bt[i] = MathF.Exp(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            for (int i = 0; i < vocabularySize; i++)
            {
                probs_bt[i] /= sum;
            }
        });
    }

    public unsafe static void CrossEntropyForward(float* losses,
                              float* probs, int* targets,
                              int batchSize, int tokenCount, int vocabularySize)
    {
        // output: losses is (batchSize,tokenCount) of the individual losses at each position
        // input: probs are (batchSize,tokenCount,vocabularySize) of the probabilities
        // input: targets is (batchSize,tokenCount) of integers giving the correct index in logits
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                // loss = -log(probs[target])
                float* probs_bt = probs + b * tokenCount * vocabularySize + t * vocabularySize;
                int ix = targets[b * tokenCount + t];
                losses[b * tokenCount + t] = -MathF.Log(probs_bt[ix]);
            }
        }
    }

    public unsafe static void CrossEntropySoftmaxBackward(float* dlogits,
                               float* dlosses, float* probs, int* targets,
                               int batchSize, int tokenCount, int vocabularySize)
    {
        // backwards through both softmax and crossentropy
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                float* dlogits_bt = dlogits + b * tokenCount * vocabularySize + t * vocabularySize;
                float* probs_bt = probs + b * tokenCount * vocabularySize + t * vocabularySize;
                float dloss = dlosses[b * tokenCount + t];
                int ix = targets[b * tokenCount + t];
                for (int i = 0; i < vocabularySize; i++)
                {
                    float p = probs_bt[i];
                    float indicator = i == ix ? 1.0f : 0.0f;
                    dlogits_bt[i] += (p - indicator) * dloss;
                }
            }
        }
    }
}
