using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

// all the individual layers' forward and backward passes
// batchSize = B, tokenCount = T, channelCount = C, vocabularySize = V
public static partial class Llm
{
    // Calling this "Encoder" is confusing as the other part of transformer
    // architecture is an encoder. This should be renamed to "Embed".
    public unsafe static void EncoderForward(float* output,
                       int* input, float* wte, float* wpe,
                       int batchSize, int tokenCount, int channelCount)
    {
        // output is (batchSize,tokenCount,channelCount). At each position (b,t), a channelCount-dimensional vector summarizing token & position
        // input is (batchSize,tokenCount) of integers, holding the token ids at each (b,t) position
        // wte is (vocabularySize,channelCount) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,channelCount) of position embeddings, short for "weight positional embedding"
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                // seek to the output position in output[b,t,:]
                float* output_bt = output + b * tokenCount * channelCount + t * channelCount;
                // get the index of the token at input[b, t]
                int ix = input[b * tokenCount + t];
                // seek to the position in wte corresponding to the token
                float* wte_ix = wte + ix * channelCount;
                // seek to the position in wpe corresponding to the position
                float* wpe_t = wpe + t * channelCount;
                // add the two vectors and store the result in output[b,t,:]
                for (int i = 0; i < channelCount; i++)
                {
                    output_bt[i] = wte_ix[i] + wpe_t[i];
                }
            }
        }
    }

    public unsafe static void EncoderBackward(float* dwte, float* dwpe,
                          float* doutput, int* input,
                          int batchSize, int tokenCount, int channelCount)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                float* doutput_bt = doutput + b * tokenCount * channelCount + t * channelCount;
                int ix = input[b * tokenCount + t];
                float* dwte_ix = dwte + ix * channelCount;
                float* dwpe_t = dwpe + t * channelCount;
                for (int i = 0; i < channelCount; i++)
                {
                    float d = doutput_bt[i];
                    dwte_ix[i] += d;
                    dwpe_t[i] += d;
                }
            }
        }
    }

    public unsafe static void LayerNormForward(float* output, float* mean, float* rstd,
                           float* input, float* weight, float* bias,
                           int batchSize, int tokenCount, int channelCount)
    {
        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both input and output are (batchSize,tokenCount,channelCount) of the activations
        // mean and rstd are (batchSize,tokenCount) buffers, to be used later in backward pass
        // at each position (b,t) of the input, the channelCount-dimensional vector
        // of activations gets normalized, then scaled and shifted
        float eps = 1e-5f;
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                // seek to the input position input[b,t,:]
                float* x = input + b * tokenCount * channelCount + t * channelCount;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0; i < channelCount; i++)
                {
                    m += x[i];
                }
                m = m / channelCount;
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0; i < channelCount; i++)
                {
                    float xshift = x[i] - m;
                    v += xshift * xshift;
                }
                v = v / channelCount;
                // calculate the rstd (reciprocal standard deviation)
                float s = 1.0f / MathF.Sqrt(v + eps);
                // seek to the output position in output[b,t,:]
                float* output_bt = output + b * tokenCount * channelCount + t * channelCount;
                for (int i = 0; i < channelCount; i++)
                {
                    float n = (s * (x[i] - m)); // normalize
                    float o = n * weight[i] + bias[i]; // scale and shift
                    output_bt[i] = o; // write
                }
                // cache the mean and rstd for the backward pass later
                mean[b * tokenCount + t] = m;
                rstd[b * tokenCount + t] = s;
            }
        }
    }

    public unsafe static void LayerNormBackward(float* dinput, float* dweight, float* dbias,
                            float* doutput, float* input, float* weight, float* mean, float* rstd,
                            int batchSize, int tokenCount, int channelCount)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                float* doutput_bt = doutput + b * tokenCount * channelCount + t * channelCount;
                float* input_bt = input + b * tokenCount * channelCount + t * channelCount;
                float* dinput_bt = dinput + b * tokenCount * channelCount + t * channelCount;
                float mean_bt = mean[b * tokenCount + t];
                float rstd_bt = rstd[b * tokenCount + t];

                // first: two reduce operations
                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int i = 0; i < channelCount; i++)
                {
                    float norm_bti = (input_bt[i] - mean_bt) * rstd_bt;
                    float dnorm_i = weight[i] * doutput_bt[i];
                    dnorm_mean += dnorm_i;
                    dnorm_norm_mean += dnorm_i * norm_bti;
                }
                dnorm_mean = dnorm_mean / channelCount;
                dnorm_norm_mean = dnorm_norm_mean / channelCount;

                // now iterate again and accumulate all the gradients
                for (int i = 0; i < channelCount; i++)
                {
                    float norm_bti = (input_bt[i] - mean_bt) * rstd_bt;
                    float dnorm_i = weight[i] * doutput_bt[i];
                    // gradient contribution to bias
                    dbias[i] += doutput_bt[i];
                    // gradient contribution to weight
                    dweight[i] += norm_bti * doutput_bt[i];
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinput_bt[i] += dval;
                }
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
        //(int b;
        //#pragma omp parallel for collapse(2)
        Parallel.ForEach(Extensions.Enumerate(batchSize, tokenCount), tuple =>
        {
            var (b, t) = tuple;
            float* output_bt = output + b * tokenCount * outputChannelCount + t * outputChannelCount;
            float* input_bt = input + b * tokenCount * inputChannelCount + t * inputChannelCount;
            for (int o = 0; o < outputChannelCount; o++)
            {
                float val = (bias != null) ? bias[o] : 0.0f;
                float* wrow = weight + o * inputChannelCount;
                for (int i = 0; i < inputChannelCount; i++)
                {
                    val += input_bt[i] * wrow[i];
                }
                output_bt[o] = val;
            }
        });
    }

    public unsafe static void MatMulBackward(float* dinput, float* dweight, float* dbias,
                         float* doutput, float* input, float* weight,
                         int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount)
    {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy

        // backward into input first, parallelize over batchSize,tokenCount
        //#pragma omp parallel for collapse(2)
        Parallel.ForEach(Extensions.Enumerate(batchSize, tokenCount), tuple =>
        {
            var (b, t) = tuple;
            float* doutput_bt = doutput + b * tokenCount * outputChannelCount + t * outputChannelCount;
            float* dinput_bt = dinput + b * tokenCount * inputChannelCount + t * inputChannelCount;
            for (int o = 0; o < outputChannelCount; o++)
            {
                float* wrow = weight + o * inputChannelCount;
                float d = doutput_bt[o];
                for (int i = 0; i < inputChannelCount; i++)
                {
                    dinput_bt[i] += wrow[i] * d;
                }
            }
        });
        // backward into weight/bias, parallelize over output channels OC
        //#pragma omp parallel for
        Parallel.For(0, outputChannelCount, o =>
        {
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < tokenCount; t++)
                {
                    float* doutput_bt = doutput + b * tokenCount * outputChannelCount + t * outputChannelCount;
                    float* input_bt = input + b * tokenCount * inputChannelCount + t * inputChannelCount;
                    float* dwrow = dweight + o * inputChannelCount;
                    float d = doutput_bt[o];
                    if (dbias != null) { dbias[o] += d; }
                    for (int i = 0; i < inputChannelCount; i++)
                    {
                        dwrow[i] += input_bt[i] * d;
                    }
                }
            }
        });
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
        Parallel.ForEach(Extensions.Enumerate(batchSize, tokenCount, headCount), tuple =>
        {
            var (b, t, h) = tuple;

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
        });
    }

    public unsafe static void AttentionBackward(float* dinput, float* dpreatt, float* datt,
                            float* doutput, float* input, float* att,
                            int batchSize, int tokenCount, int channelCount, int headCount)
    {
        // input/dinput are (batchSize, tokenCount, 3C) Q,K,vocabularySize
        // att/datt/dpreatt are (batchSize, headCount, tokenCount, tokenCount)
        // doutput is (batchSize, tokenCount, channelCount)
        int C3 = channelCount * 3;
        int headSize = channelCount / headCount; // head size
        float scale = 1.0f / MathF.Sqrt(headSize);

        for (int b = 0; b < batchSize; b++)
        {
            for (int t = 0; t < tokenCount; t++)
            {
                for (int h = 0; h < headCount; h++)
                {
                    float* att_bth = att + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
                    float* datt_bth = datt + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
                    float* dpreatt_bth = dpreatt + b * headCount * tokenCount * tokenCount + h * tokenCount * tokenCount + t * tokenCount;
                    float* dquery_t = dinput + b * tokenCount * C3 + t * C3 + h * headSize;
                    float* query_t = input + b * tokenCount * C3 + t * C3 + h * headSize;

                    // backward pass 4, through the value accumulation
                    float* doutput_bth = doutput + b * tokenCount * channelCount + t * channelCount + h * headSize;
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float* value_t2 = input + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount * 2; // +channelCount*2 because it's value
                        float* dvalue_t2 = dinput + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount * 2;
                        for (int i = 0; i < headSize; i++)
                        {
                            // in the forward pass this was:
                            // output_bth[i] += att_bth[t2] * value_t2[i];
                            // so now we have:
                            datt_bth[t2] += value_t2[i] * doutput_bth[i];
                            dvalue_t2[i] += att_bth[t2] * doutput_bth[i];
                        }
                    }

                    // backward pass 2 & 3, the softmax
                    // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        for (int t3 = 0; t3 <= t; t3++)
                        {
                            float indicator = t2 == t3 ? 1.0f : 0.0f;
                            float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                            dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                        }
                    }

                    // backward pass 1, the query @ key matmul
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float* key_t2 = input + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount; // +channelCount because it's key
                        float* dkey_t2 = dinput + b * tokenCount * C3 + t2 * C3 + h * headSize + channelCount; // +channelCount because it's key
                        for (int i = 0; i < headSize; i++)
                        {
                            // in the forward pass this was:
                            // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                            // so now we have:
                            dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                            dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                        }
                    }
                }
            }
        }
    }

    static readonly float GELU_SCALING_FACTOR = MathF.Sqrt(2.0f / MathF.PI);
    public unsafe static void GeLUForward(float* output, float* input, int count)
    {
        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        for (int i = 0; i < count; i++)
        {
            float x = input[i];
            float cube = 0.044715f * x * x * x;
            output[i] = 0.5f * x * (1.0f + MathF.Tanh(GELU_SCALING_FACTOR * (x + cube)));
        }
    }

    public unsafe static void GeLUBackward(float* dinput, float* input, float* doutput, int count)
    {
        for (int i = 0; i < count; i++)
        {
            float x = input[i];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = MathF.Tanh(tanh_arg);
            float coshf_out = MathF.Cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            dinput[i] += local_grad * doutput[i];
        }
    }

    public unsafe static void ResidualForward(float* output, float* input1, float* input2, int count)
    {
        for (int i = 0; i < count; i++)
        {
            output[i] = input1[i] + input2[i];
        }
    }

    public unsafe static void ResidualBackward(float* dinput1, float* dinput2, float* doutput, int count)
    {
        for (int i = 0; i < count; i++)
        {
            dinput1[i] += doutput[i];
            dinput2[i] += doutput[i];
        }
    }

    public unsafe static void SoftmaxForward(float* probs, float* logits, int batchSize, int tokenCount, int vocabularySize)
    {
        // output: probs are (batchSize,tokenCount,vocabularySize) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (batchSize,tokenCount,vocabularySize) of the unnormalized log probabilities
        //#pragma omp parallel for collapse(2)
        //for (b = 0; b < batchSize; b++)
        Parallel.ForEach(Extensions.Enumerate(batchSize, tokenCount), tuple =>
        {
            var (b, t) = tuple;
            // probs <- softmax(logits)
            float* logits_bt = logits + b * tokenCount * vocabularySize + t * vocabularySize;
            float* probs_bt = probs + b * tokenCount * vocabularySize + t * vocabularySize;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
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
