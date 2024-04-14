using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace nietras.LargeLanguageModel;

public static partial class Gpt2
{
    static readonly Action<string> Log = t => { Console.WriteLine(t); Trace.WriteLine(t); };

    // ----------------------------------------------------------------------------
    // all the individual layers' forward and backward passes
    // B = batch_size, T = sequence_length, C = channels, V = vocab_size

    public unsafe static void encoder_forward(float* output,
                       int* inp, float* wte, float* wpe,
                       int B, int T, int C)
    {
        // output is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        // inp is (B,T) of integers, holding the token ids at each (b,t) position
        // wte is (V,C) of token embeddings, short for "weight token embeddings"
        // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // seek to the output position in output[b,t,:]
                float* out_bt = output + b * T * C + t * C;
                // get the index of the token at inp[b, t]
                int ix = inp[b * T + t];
                // seek to the position in wte corresponding to the token
                float* wte_ix = wte + ix * C;
                // seek to the position in wpe corresponding to the position
                float* wpe_t = wpe + t * C;
                // add the two vectors and store the result in output[b,t,:]
                for (int i = 0; i < C; i++)
                {
                    out_bt[i] = wte_ix[i] + wpe_t[i];
                }
            }
        }
    }

    public unsafe static void encoder_backward(float* dwte, float* dwpe,
                          float* dout, int* inp,
                          int B, int T, int C)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float* dout_bt = dout + b * T * C + t * C;
                int ix = inp[b * T + t];
                float* dwte_ix = dwte + ix * C;
                float* dwpe_t = dwpe + t * C;
                for (int i = 0; i < C; i++)
                {
                    float d = dout_bt[i];
                    dwte_ix[i] += d;
                    dwpe_t[i] += d;
                }
            }
        }
    }

    public unsafe static void layernorm_forward(float* output, float* mean, float* rstd,
                           float* inp, float* weight, float* bias,
                           int B, int T, int C)
    {
        // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        // both inp and output are (B,T,C) of the activations
        // mean and rstd are (B,T) buffers, to be used later in backward pass
        // at each position (b,t) of the input, the C-dimensional vector
        // of activations gets normalized, then scaled and shifted
        float eps = 1e-5f;
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // seek to the input position inp[b,t,:]
                float* x = inp + b * T * C + t * C;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    m += x[i];
                }
                m = m / C;
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    float xshift = x[i] - m;
                    v += xshift * xshift;
                }
                v = v / C;
                // calculate the rstd (reciprocal standard deviation)
                float s = 1.0f / MathF.Sqrt(v + eps);
                // seek to the output position in output[b,t,:]
                float* out_bt = output + b * T * C + t * C;
                for (int i = 0; i < C; i++)
                {
                    float n = (s * (x[i] - m)); // normalize
                    float o = n * weight[i] + bias[i]; // scale and shift
                    out_bt[i] = o; // write
                }
                // cache the mean and rstd for the backward pass later
                mean[b * T + t] = m;
                rstd[b * T + t] = s;
            }
        }
    }

    public unsafe static void layernorm_backward(float* dinp, float* dweight, float* dbias,
                            float* dout, float* inp, float* weight, float* mean, float* rstd,
                            int B, int T, int C)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float* dout_bt = dout + b * T * C + t * C;
                float* inp_bt = inp + b * T * C + t * C;
                float* dinp_bt = dinp + b * T * C + t * C;
                float mean_bt = mean[b * T + t];
                float rstd_bt = rstd[b * T + t];

                // first: two reduce operations
                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int i = 0; i < C; i++)
                {
                    float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                    float dnorm_i = weight[i] * dout_bt[i];
                    dnorm_mean += dnorm_i;
                    dnorm_norm_mean += dnorm_i * norm_bti;
                }
                dnorm_mean = dnorm_mean / C;
                dnorm_norm_mean = dnorm_norm_mean / C;

                // now iterate again and accumulate all the gradients
                for (int i = 0; i < C; i++)
                {
                    float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                    float dnorm_i = weight[i] * dout_bt[i];
                    // gradient contribution to bias
                    dbias[i] += dout_bt[i];
                    // gradient contribution to weight
                    dweight[i] += norm_bti * dout_bt[i];
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i; // term 1
                    dval -= dnorm_mean; // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt; // final scale
                    dinp_bt[i] += dval;
                }
            }
        }
    }

    public unsafe static void matmul_forward(float* output,
                        float* inp, float* weight, float* bias,
                        int B, int T, int C, int OC)
    {
        // most of the running time is spent here and in matmul_backward
        // OC is short for "output channels"
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        // output will be (B,T,OC)
        int b;
#pragma omp parallel for collapse(2)
        for (b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float* out_bt = output + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                for (int o = 0; o < OC; o++)
                {
                    float val = (bias != null) ? bias[o] : 0.0f;
                    float* wrow = weight + o * C;
                    for (int i = 0; i < C; i++)
                    {
                        val += inp_bt[i] * wrow[i];
                    }
                    out_bt[o] = val;
                }
            }
        }
    }

    public unsafe static void matmul_backward(float* dinp, float* dweight, float* dbias,
                         float* dout, float* inp, float* weight,
                         int B, int T, int C, int OC)
    {
        // most of the running time is spent here and in matmul_forward
        // this backward could be done in a single "round" of loops
        // but that doesn't afford an efficient parallelization strategy

        // backward into inp first, parallelize over B,T
#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* dinp_bt = dinp + b * T * C + t * C;
                for (int o = 0; o < OC; o++)
                {
                    float* wrow = weight + o * C;
                    float d = dout_bt[o];
                    for (int i = 0; i < C; i++)
                    {
                        dinp_bt[i] += wrow[i] * d;
                    }
                }
            }
        }
        // backward into weight/bias, parallelize over output channels OC
#pragma omp parallel for
        for (int o = 0; o < OC; o++)
        {
            for (int b = 0; b < B; b++)
            {
                for (int t = 0; t < T; t++)
                {
                    float* dout_bt = dout + b * T * OC + t * OC;
                    float* inp_bt = inp + b * T * C + t * C;
                    float* dwrow = dweight + o * C;
                    float d = dout_bt[o];
                    if (dbias != null) { dbias[o] += d; }
                    for (int i = 0; i < C; i++)
                    {
                        dwrow[i] += inp_bt[i] * d;
                    }
                }
            }
        }
    }

    public unsafe static void attention_forward(float* output, float* preatt, float* att,
                           float* inp,
                           int B, int T, int C, int NH)
    {
        // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
        // that holds the pre-attention and post-attention scores (used in backward)
        // output is (B, T, C)
        // attention is the only layer that mixes information across time
        // every other operation is applied at every (b,t) position independently
        // (and of course, no layer mixes information across batch)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / MathF.Sqrt(hs);

        int b;
#pragma omp parallel for collapse(3)
        for (b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                for (int h = 0; h < NH; h++)
                {
                    float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                    float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                    float* att_bth = att + b * NH * T * T + h * T * T + t * T;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++)
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
                    for (int t2 = 0; t2 < T; t2++)
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
                    float* out_bth = output + b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        float att_btht2 = att_bth[t2];
                        for (int i = 0; i < hs; i++)
                        {
                            out_bth[i] += att_btht2 * value_t2[i];
                        }
                    }
                }
            }
        }
    }

    public unsafe static void attention_backward(float* dinp, float* dpreatt, float* datt,
                            float* dout, float* inp, float* att,
                            int B, int T, int C, int NH)
    {
        // inp/dinp are (B, T, 3C) Q,K,V
        // att/datt/dpreatt are (B, NH, T, T)
        // dout is (B, T, C)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / MathF.Sqrt(hs);

        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                for (int h = 0; h < NH; h++)
                {
                    float* att_bth = att + b * NH * T * T + h * T * T + t * T;
                    float* datt_bth = datt + b * NH * T * T + h * T * T + t * T;
                    float* dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                    float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                    float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                    // backward pass 4, through the value accumulation
                    float* dout_bth = dout + b * T * C + t * C + h * hs;
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                        for (int i = 0; i < hs; i++)
                        {
                            // in the forward pass this was:
                            // out_bth[i] += att_bth[t2] * value_t2[i];
                            // so now we have:
                            datt_bth[t2] += value_t2[i] * dout_bth[i];
                            dvalue_t2[i] += att_bth[t2] * dout_bth[i];
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
                        float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                        float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                        for (int i = 0; i < hs; i++)
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
    public unsafe static void gelu_forward(float* output, float* inp, int N)
    {
        // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
        for (int i = 0; i < N; i++)
        {
            float x = inp[i];
            float cube = 0.044715f * x * x * x;
            output[i] = 0.5f * x * (1.0f + MathF.Tanh(GELU_SCALING_FACTOR * (x + cube)));
        }
    }

    public unsafe static void gelu_backward(float* dinp, float* inp, float* dout, int N)
    {
        for (int i = 0; i < N; i++)
        {
            float x = inp[i];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
            float tanh_out = MathF.Tanh(tanh_arg);
            float coshf_out = MathF.Cosh(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
            dinp[i] += local_grad * dout[i];
        }
    }

    public unsafe static void residual_forward(float* output, float* inp1, float* inp2, int N)
    {
        for (int i = 0; i < N; i++)
        {
            output[i] = inp1[i] + inp2[i];
        }
    }

    public unsafe static void residual_backward(float* dinp1, float* dinp2, float* dout, int N)
    {
        for (int i = 0; i < N; i++)
        {
            dinp1[i] += dout[i];
            dinp2[i] += dout[i];
        }
    }

    public unsafe static void softmax_forward(float* probs, float* logits, int B, int T, int V)
    {
        // output: probs are (B,T,V) of the probabilities (sums to 1.0 in each b,t position)
        // input: logits is (B,T,V) of the unnormalized log probabilities
        int b;
#pragma omp parallel for collapse(2)
        for (b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // probs <- softmax(logits)
                float* logits_bt = logits + b * T * V + t * V;
                float* probs_bt = probs + b * T * V + t * V;

                // maxval is only calculated and subtracted for numerical stability
                float maxval = -10000.0f; // TODO something better
                for (int i = 0; i < V; i++)
                {
                    if (logits_bt[i] > maxval)
                    {
                        maxval = logits_bt[i];
                    }
                }
                float sum = 0.0f;
                for (int i = 0; i < V; i++)
                {
                    probs_bt[i] = MathF.Exp(logits_bt[i] - maxval);
                    sum += probs_bt[i];
                }
                for (int i = 0; i < V; i++)
                {
                    probs_bt[i] /= sum;
                }
            }
        }
    }

    public unsafe static void crossentropy_forward(float* losses,
                              float* probs, int* targets,
                              int B, int T, int V)
    {
        // output: losses is (B,T) of the individual losses at each position
        // input: probs are (B,T,V) of the probabilities
        // input: targets is (B,T) of integers giving the correct index in logits
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // loss = -log(probs[target])
                float* probs_bt = probs + b * T * V + t * V;
                int ix = targets[b * T + t];
                losses[b * T + t] = -MathF.Log(probs_bt[ix]);
            }
        }
    }

    public unsafe static void crossentropy_softmax_backward(float* dlogits,
                               float* dlosses, float* probs, int* targets,
                               int B, int T, int V)
    {
        // backwards through both softmax and crossentropy
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float* dlogits_bt = dlogits + b * T * V + t * V;
                float* probs_bt = probs + b * T * V + t * V;
                float dloss = dlosses[b * T + t];
                int ix = targets[b * T + t];
                for (int i = 0; i < V; i++)
                {
                    float p = probs_bt[i];
                    float indicator = i == ix ? 1.0f : 0.0f;
                    dlogits_bt[i] += (p - indicator) * dloss;
                }
            }
        }
    }

    // ----------------------------------------------------------------------------
    // GPT-2 model definition

    // the parameters of the model
    const int NUM_PARAMETER_TENSORS = 16;
    public unsafe struct ParameterTensors
    {
        public float* wte; // (V, C)
        public float* wpe; // (maxT, C)
        public float* ln1w; // (L, C)
        public float* ln1b; // (L, C)
        public float* qkvw; // (L, 3*C, C)
        public float* qkvb; // (L, 3*C)
        public float* attprojw; // (L, C, C)
        public float* attprojb; // (L, C)
        public float* ln2w; // (L, C)
        public float* ln2b; // (L, C)
        public float* fcw; // (L, 4*C, C)
        public float* fcb; // (L, 4*C)
        public float* fcprojw; // (L, C, 4*C)
        public float* fcprojb; // (L, C)
        public float* lnfw; // (C)
        public float* lnfb; // (C)
    }

    // allocate memory for the parameters and point the individual tensors to the right places
    public unsafe static float* malloc_and_point_parameters(ParameterTensors* parameters, long* param_sizes)
    {
        long num_parameters = 0;
        for (long i = 0; i < NUM_PARAMETER_TENSORS; i++)
        {
            num_parameters += param_sizes[i];
        }
        // malloc all parameters all at once
        float* params_memory = malloc<float>(num_parameters);
        // assign all the tensors
        float**[] ptrs = [
        &parameters->wte, &parameters->wpe, &parameters->ln1w, &parameters->ln1b, &parameters->qkvw, &parameters->qkvb,
        &parameters->attprojw, &parameters->attprojb, &parameters->ln2w, &parameters->ln2b, &parameters->fcw, &parameters->fcb,
        &parameters->fcprojw, &parameters->fcprojb, &parameters->lnfw, &parameters->lnfb
    ];
        float* params_memory_iterator = params_memory;
        for (long i = 0; i < NUM_PARAMETER_TENSORS; i++)
        {
            *(ptrs[i]) = params_memory_iterator;
            params_memory_iterator += param_sizes[i];
        }
        return params_memory;
    }

    const int NUM_ACTIVATION_TENSORS = 23;
    public unsafe struct ActivationTensors
    {
        public float* encoded; // (B, T, C)
        public float* ln1; // (L, B, T, C)
        public float* ln1_mean; // (L, B, T)
        public float* ln1_rstd; // (L, B, T)
        public float* qkv; // (L, B, T, 3*C)
        public float* atty; // (L, B, T, C)
        public float* preatt; // (L, B, NH, T, T)
        public float* att; // (L, B, NH, T, T)
        public float* attproj; // (L, B, T, C)
        public float* residual2; // (L, B, T, C)
        public float* ln2; // (L, B, T, C)
        public float* ln2_mean; // (L, B, T)
        public float* ln2_rstd; // (L, B, T)
        public float* fch; // (L, B, T, 4*C)
        public float* fch_gelu; // (L, B, T, 4*C)
        public float* fcproj; // (L, B, T, C)
        public float* residual3; // (L, B, T, C)
        public float* lnf; // (B, T, C)
        public float* lnf_mean; // (B, T)
        public float* lnf_rstd; // (B, T)
        public float* logits; // (B, T, V)
        public float* probs; // (B, T, V)
        public float* losses; // (B, T)
    }

    public unsafe static float* malloc_and_point_activations(ActivationTensors* acts, long* act_sizes)
    {
        long num_activations = 0;
        for (long i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        {
            num_activations += act_sizes[i];
        }
        float* acts_memory = malloc<float>(num_activations);
        float**[] ptrs = [
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    ];
        float* acts_memory_iterator = acts_memory;
        for (long i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        {
            *(ptrs[i]) = acts_memory_iterator;
            acts_memory_iterator += act_sizes[i];
        }
        return acts_memory;
    }

    public unsafe struct GPT2Config
    {
        public int max_seq_len; // max sequence length, e.g. 1024
        public int vocab_size; // vocab size, e.g. 50257
        public int num_layers; // number of layers, e.g. 12
        public int num_heads; // number of heads in attention, e.g. 12
        public int channels; // number of channels, e.g. 768
    }

    public unsafe struct GPT2
    {
        public GPT2Config config;
        // the weights (parameters) of the model, and their sizes
        public ParameterTensors parameters;
        public fixed long param_sizes[NUM_PARAMETER_TENSORS];
        public float* params_memory;
        public long num_parameters;
        // gradients of the weights
        public ParameterTensors grads;
        public float* grads_memory;
        // buffers for the AdamW optimizer
        public float* m_memory;
        public float* v_memory;
        // the activations of the model, and their sizes
        public ActivationTensors acts;
        public fixed long act_sizes[NUM_ACTIVATION_TENSORS];
        public float* acts_memory;
        public long num_activations;
        // gradients of the activations
        public ActivationTensors grads_acts;
        public float* grads_acts_memory;
        // other run state configuration
        public int batch_size; // the batch size (B) of current forward pass
        public int seq_len; // the sequence length (T) of current forward pass
        public int* inputs; // the input tokens for the current forward pass
        public int* targets; // the target tokens for the current forward pass
        public float mean_loss; // after a forward pass with targets, will be populated with the mean loss
    }

    public unsafe static void gpt2_build_from_checkpoint(GPT2* model, string checkpoint_path)
    {
        // read in model from a checkpoint file
        using var model_file = File.OpenRead(checkpoint_path);
        Span<int> model_header = stackalloc int[256];
        // read span from model_file
        model_file.ReadExactlyUnmanaged(model_header);
        //fread(model_header, sizeof(int), 256, model_file);
        if (model_header[0] != 20240326) { throw new InvalidDataException($"Bad magic model file"); }
        if (model_header[1] != 1) { throw new InvalidDataException($"Bad version in model file"); }

        // read in hyperparameters
        int maxT, V, L, NH, C;
        model->config.max_seq_len = maxT = model_header[2];
        model->config.vocab_size = V = model_header[3];
        model->config.num_layers = L = model_header[4];
        model->config.num_heads = NH = model_header[5];
        model->config.channels = C = model_header[6];
        Log("[GPT-2]");
        Log($"max_seq_len: {maxT}");
        Log($"vocab_size: {V}");
        Log($"num_layers: {L}");
        Log($"num_heads: {NH}");
        Log($"channels: {C}");

        // allocate space for all the parameters and read them in
        model->param_sizes[0] = V * C; // wte
        model->param_sizes[1] = maxT * C; // wpe
        model->param_sizes[2] = L * C; // ln1w
        model->param_sizes[3] = L * C; // ln1b
        model->param_sizes[4] = L * (3 * C) * C; // qkvw
        model->param_sizes[5] = L * (3 * C); // qkvb
        model->param_sizes[6] = L * C * C; // attprojw
        model->param_sizes[7] = L * C; // attprojb
        model->param_sizes[8] = L * C; // ln2w
        model->param_sizes[9] = L * C; // ln2b
        model->param_sizes[10] = L * (4 * C) * C; // fcw
        model->param_sizes[11] = L * (4 * C); // fcb
        model->param_sizes[12] = L * C * (4 * C); // fcprojw
        model->param_sizes[13] = L * C; // fcprojb
        model->param_sizes[14] = C; // lnfw
        model->param_sizes[15] = C; // lnfb

        // cound the number of paramaters
        long num_parameters = 0;
        for (long i = 0; i < NUM_PARAMETER_TENSORS; i++)
        {
            num_parameters += model->param_sizes[i];
        }
        Log($"num_parameters: {num_parameters}");
        model->num_parameters = num_parameters;

        // read in all the parameters from file
        model->params_memory = malloc_and_point_parameters(&model->parameters, model->param_sizes);
        ReadExactlyUnmanaged(model_file, model->params_memory, num_parameters);

        // other inits
        model->acts_memory = null;
        model->grads_memory = null;
        model->m_memory = null;
        model->v_memory = null;
        model->grads_acts_memory = null;
        model->inputs = null;
        model->targets = null;
        model->batch_size = 0;
        model->seq_len = 0;
        model->mean_loss = -1.0f; // -1.0f will designate no loss
    }

    static unsafe void gpt2_forward(GPT2* model, int* inputs, int* targets, int B, int T)
    {
        // targets are optional and could be null

        // ensure the model was initialized or error output
        if (model->params_memory == null)
        {
            throw new InvalidOperationException("Error: model was not initialized properly.");
        }

        // convenience parameters
        int V = model->config.vocab_size;
        int L = model->config.num_layers;
        int NH = model->config.num_heads;
        int C = model->config.channels;

        // allocate space for all the activations if needed (done here, lazily)
        if (model->acts_memory == null)
        {
            // record the current B,T as well
            model->batch_size = B;
            model->seq_len = T;
            // and now allocate the space
            model->act_sizes[0] = B * T * C; // encoded
            model->act_sizes[1] = L * B * T * C; // ln1
            model->act_sizes[2] = L * B * T;  // ln1_mean
            model->act_sizes[3] = L * B * T;  // ln1_rstd
            model->act_sizes[4] = L * B * T * 3 * C; // qkv
            model->act_sizes[5] = L * B * T * C;  // atty
            model->act_sizes[6] = L * B * NH * T * T;  // preatt
            model->act_sizes[7] = L * B * NH * T * T;  // att
            model->act_sizes[8] = L * B * T * C; // attproj
            model->act_sizes[9] = L * B * T * C; // residual2
            model->act_sizes[10] = L * B * T * C; // ln2
            model->act_sizes[11] = L * B * T; // ln2_mean
            model->act_sizes[12] = L * B * T; // ln2_rstd
            model->act_sizes[13] = L * B * T * 4 * C; // fch
            model->act_sizes[14] = L * B * T * 4 * C; // fch_gelu
            model->act_sizes[15] = L * B * T * C; // fcproj
            model->act_sizes[16] = L * B * T * C; // residual3
            model->act_sizes[17] = B * T * C; // lnf
            model->act_sizes[18] = B * T; // lnf_mean
            model->act_sizes[19] = B * T; // lnf_rstd
            model->act_sizes[20] = B * T * V; // logits
            model->act_sizes[21] = B * T * V; // probs
            model->act_sizes[22] = B * T; // losses
            long num_activations = 0;
            for (long i = 0; i < NUM_ACTIVATION_TENSORS; i++)
            {
                num_activations += model->act_sizes[i];
            }
            Log($"num_activations: {num_activations}");
            model->num_activations = num_activations;
            model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
            // also create memory for caching inputs and targets
            model->inputs = malloc<int>(B * T);
            model->targets = malloc<int>(B * T); // might be unused if we never have targets but it's small
        }
        else
        {
            // validate B,T is no larger than what was previously allocated
            // in principle, we could re-allocate a larger chunk of memory, for now we just error output
            if (B > model->batch_size || T > model->seq_len)
            {
                throw new InvalidDataException("Error: batch size or sequence length is inadequately large" +
                    $"Model: B={model->batch_size} T={model->seq_len}, Desired: B={B} T={T}");
            }
        }

        // cache the inputs/targets
        memcpy(model->inputs, inputs, B * T * sizeof(int));
        if (targets != null)
        {
            memcpy(model->targets, targets, B * T * sizeof(int));
        }

        // forward pass
        ParameterTensors parameters = model->parameters; // for brevity
        ActivationTensors acts = model->acts;
        float* residual;
        encoder_forward(acts.encoded, inputs, parameters.wte, parameters.wpe, B, T, C); // encoding goes into residual[0]
        for (int l = 0; l < L; l++)
        {

            residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

            // get the pointers of the weights for this layer
            float* l_ln1w = parameters.ln1w + l * C;
            float* l_ln1b = parameters.ln1b + l * C;
            float* l_qkvw = parameters.qkvw + l * 3 * C * C;
            float* l_qkvb = parameters.qkvb + l * 3 * C;
            float* l_attprojw = parameters.attprojw + l * C * C;
            float* l_attprojb = parameters.attprojb + l * C;
            float* l_ln2w = parameters.ln2w + l * C;
            float* l_ln2b = parameters.ln2b + l * C;
            float* l_fcw = parameters.fcw + l * 4 * C * C;
            float* l_fcb = parameters.fcb + l * 4 * C;
            float* l_fcprojw = parameters.fcprojw + l * C * 4 * C;
            float* l_fcprojb = parameters.fcprojb + l * C;

            // get the pointers of the activations for this layer
            float* l_ln1 = acts.ln1 + l * B * T * C;
            float* l_ln1_mean = acts.ln1_mean + l * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            float* l_qkv = acts.qkv + l * B * T * 3 * C;
            float* l_atty = acts.atty + l * B * T * C;
            float* l_preatt = acts.preatt + l * B * NH * T * T;
            float* l_att = acts.att + l * B * NH * T * T;
            float* l_attproj = acts.attproj + l * B * T * C;
            float* l_residual2 = acts.residual2 + l * B * T * C;
            float* l_ln2 = acts.ln2 + l * B * T * C;
            float* l_ln2_mean = acts.ln2_mean + l * B * T;
            float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            float* l_fch = acts.fch + l * B * T * 4 * C;
            float* l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
            float* l_fcproj = acts.fcproj + l * B * T * C;
            float* l_residual3 = acts.residual3 + l * B * T * C;

            // now do the forward pass
            layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
            matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C);
            attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
            matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
            residual_forward(l_residual2, residual, l_attproj, B * T * C);
            layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
            matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C);
            gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C);
            matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C);
            residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C);
        }
        residual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
        layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, parameters.lnfw, parameters.lnfb, B, T, C);
        matmul_forward(acts.logits, acts.lnf, parameters.wte, null, B, T, C, V);
        softmax_forward(acts.probs, acts.logits, B, T, V);

        // also forward the cross-entropy loss function if we have the targets
        if (targets != null)
        {
            crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, V);
            // for convenience also evaluate the mean loss
            float mean_loss = 0.0f;
            for (int i = 0; i < B * T; i++) { mean_loss += model->acts.losses[i]; }
            mean_loss /= B * T;
            model->mean_loss = mean_loss;
        }
        else
        {
            // if we don't have targets, we don't have a loss
            model->mean_loss = -1.0f;
        }
    }

    static unsafe void gpt2_zero_grad(GPT2* model)
    {
        if (model->grads_memory != null) { memset(model->grads_memory, model->num_parameters); }
        if (model->grads_acts_memory != null) { memset(model->grads_acts_memory, model->num_activations); }
    }

    static unsafe void gpt2_backward(GPT2* model)
    {

        // double check we forwarded previously, with targets
        if (model->mean_loss == -1.0f)
        {
            throw new InvalidOperationException("Error: must forward with targets before backward");
        }

        // lazily allocate the memory for gradients of the weights and activations, if needed
        if (model->grads_memory == null)
        {
            model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
            model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
            gpt2_zero_grad(model);
        }

        // convenience shortcuts
        int B = model->batch_size;
        int T = model->seq_len;
        int V = model->config.vocab_size;
        int L = model->config.num_layers;
        int NH = model->config.num_heads;
        int C = model->config.channels;

        // backward pass: go in the reverse order of the forward pass, and call backward() functions
        ParameterTensors parameters = model->parameters; // for brevity
        ParameterTensors grads = model->grads;
        ActivationTensors acts = model->acts;
        ActivationTensors grads_acts = model->grads_acts;

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        float dloss_mean = 1.0f / (B * T);
        for (int i = 0; i < B * T; i++) { grads_acts.losses[i] = dloss_mean; }

        crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V);
        matmul_backward(grads_acts.lnf, grads.wte, null, grads_acts.logits, acts.lnf, parameters.wte, B, T, C, V);
        float* residual = acts.residual3 + (L - 1) * B * T * C; // last layer's residual
        float* dresidual = grads_acts.residual3 + (L - 1) * B * T * C; // write to last layer's residual
        layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, parameters.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

        for (int l = L - 1; l >= 0; l--)
        {

            residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
            dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l - 1) * B * T * C;

            // get the pointers of the weights for this layer
            float* l_ln1w = parameters.ln1w + l * C;
            float* l_qkvw = parameters.qkvw + l * 3 * C * C;
            float* l_attprojw = parameters.attprojw + l * C * C;
            float* l_ln2w = parameters.ln2w + l * C;
            float* l_fcw = parameters.fcw + l * 4 * C * C;
            float* l_fcprojw = parameters.fcprojw + l * C * 4 * C;
            // get the pointers of the gradients of the weights for this layer
            float* dl_ln1w = grads.ln1w + l * C;
            float* dl_ln1b = grads.ln1b + l * C;
            float* dl_qkvw = grads.qkvw + l * 3 * C * C;
            float* dl_qkvb = grads.qkvb + l * 3 * C;
            float* dl_attprojw = grads.attprojw + l * C * C;
            float* dl_attprojb = grads.attprojb + l * C;
            float* dl_ln2w = grads.ln2w + l * C;
            float* dl_ln2b = grads.ln2b + l * C;
            float* dl_fcw = grads.fcw + l * 4 * C * C;
            float* dl_fcb = grads.fcb + l * 4 * C;
            float* dl_fcprojw = grads.fcprojw + l * C * 4 * C;
            float* dl_fcprojb = grads.fcprojb + l * C;
            // get the pointers of the activations for this layer
            float* l_ln1 = acts.ln1 + l * B * T * C;
            float* l_ln1_mean = acts.ln1_mean + l * B * T;
            float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
            float* l_qkv = acts.qkv + l * B * T * 3 * C;
            float* l_atty = acts.atty + l * B * T * C;
            float* l_att = acts.att + l * B * NH * T * T;
            float* l_residual2 = acts.residual2 + l * B * T * C;
            float* l_ln2 = acts.ln2 + l * B * T * C;
            float* l_ln2_mean = acts.ln2_mean + l * B * T;
            float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
            float* l_fch = acts.fch + l * B * T * 4 * C;
            float* l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C;
            // get the pointers of the gradients of the activations for this layer
            float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
            float* dl_qkv = grads_acts.qkv + l * B * T * 3 * C;
            float* dl_atty = grads_acts.atty + l * B * T * C;
            float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
            float* dl_att = grads_acts.att + l * B * NH * T * T;
            float* dl_attproj = grads_acts.attproj + l * B * T * C;
            float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
            float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
            float* dl_fch = grads_acts.fch + l * B * T * 4 * C;
            float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4 * C;
            float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
            float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

            // backprop this layer
            residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C);
            matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C);
            gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C);
            matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C);
            layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
            residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C);
            matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
            attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
            matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C);
            layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
        }
        encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
    }

    static unsafe void gpt2_update(GPT2* model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t)
    {
        // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

        // lazily allocate the memory for m_memory and v_memory
        if (model->m_memory == null)
        {
            model->m_memory = calloc<float>(model->num_parameters);
            model->v_memory = calloc<float>(model->num_parameters);
        }

        for (int i = 0; i < model->num_parameters; i++)
        {
            float param = model->params_memory[i];
            float grad = model->grads_memory[i];

            // update the first moment (momentum)
            float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
            // update the second moment (RMSprop)
            float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
            // bias-correct both moments
            float m_hat = m / (1.0f - MathF.Pow(beta1, t));
            float v_hat = v / (1.0f - MathF.Pow(beta2, t));

            // update
            model->m_memory[i] = m;
            model->v_memory[i] = v;
            model->params_memory[i] -= learning_rate * (m_hat / (MathF.Sqrt(v_hat) + eps) + weight_decay * param);
        }
    }

    static unsafe void gpt2_free(GPT2* model)
    {
        free(model->params_memory);
        free(model->grads_memory);
        free(model->m_memory);
        free(model->v_memory);
        free(model->acts_memory);
        free(model->grads_acts_memory);
        free(model->inputs);
        free(model->targets);
    }

    // ----------------------------------------------------------------------------
    // data loader lite
    // returns random batches of data from a file of integers

    public unsafe class DataLoader : IDisposable
    {
        // hyperparameters
        public int B; // batch size
        public int T; // sequence length
                      // input handling and its state
        public FileStream tokens_file;
        public long file_size;
        public long current_position;
        // output memory
        public int* batch;
        public int* inputs;
        public int* targets;
        // convenience variables
        public long num_batches;
        bool _disposedValue;

        public DataLoader(string filename, int B, int T)
        {
            this.B = B;
            this.T = T;

            // open the input file for reading
            this.tokens_file = File.OpenRead(filename);
            this.file_size = tokens_file.Length;
            if (this.file_size < (B * T + 1) * sizeof(int))
            {
                throw new InvalidDataException($"Error: file size is too small for the batch size and sequence length");
            }

            // allocate space for B*T + 1 integers to store the inputs and targets
            this.batch = malloc<int>((B * T + 1));
            this.inputs = this.batch;
            this.targets = this.batch + 1; // targets are shifted by one
            this.num_batches = this.file_size / (B * T * sizeof(int));
        }

        public unsafe void dataloader_reset()
        {
            this.tokens_file.Position = 0;
        }

        public unsafe void dataloader_next_batch()
        {
            // if we are at the end of the file, loop back to the beginning
            if (this.tokens_file.Position + (B * T + 1) * sizeof(int) > this.file_size)
            {
                this.tokens_file.Position = 0;
            }
            // read the B*T+1 integers from the file into batch
            tokens_file.ReadExactlyUnmanaged(this.batch, B * T + 1);
            //fread(this.batch, sizeof(int), B * T + 1, this.tokens_file);
            // advance the current position by B*T integers 
            //this.current_position += B * T * sizeof(int);
            // Read +1 more token to get the target and hence have to move back
            tokens_file.Position -= sizeof(int);
        }

        public unsafe void dataloader_free()
        {
            Dispose(true);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    this.tokens_file.Dispose();
                }
                free(this.batch);
                _disposedValue = true;
            }
        }

        // // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        // ~DataLoader()
        // {
        //     // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
        //     Dispose(disposing: false);
        // }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }


    // Add the following code to the Llm class

    public unsafe static T* calloc<T>(long size) where T : unmanaged
    {
        var ptr = malloc<T>(size);
        memset(ptr, size);
        return ptr;
    }

    public unsafe static T* malloc<T>(long size) where T : unmanaged
    {
        return (T*)NativeMemory.Alloc((nuint)(size * sizeof(T)));
    }

    public unsafe static void free<T>(T* ptr) where T : unmanaged
    {
        NativeMemory.Free(ptr);
    }

    public unsafe static void memcpy<T>(T* dest, T* src, long size) where T : unmanaged
    {
        var sizeInBytes = size * sizeof(T);
        Buffer.MemoryCopy(src, dest, sizeInBytes, sizeInBytes);
    }

    public unsafe static void memset<T>(T* ptr, long size) where T : unmanaged
    {
        NativeMemory.Clear(ptr, (nuint)(size * sizeof(T)));
    }


    // ----------------------------------------------------------------------------
    // sampler

    // the GPT-2 end-of-text token id
    const int GPT2_EOT = 50256;

    static unsafe uint random_u32(ulong* state)
    {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (uint)((*state * 0x2545F4914F6CDD1Dul) >> 32);
    }
    static unsafe float random_f32(ulong* state)
    { // random float32 in [0,1)
        return (random_u32(state) >> 8) / 16777216.0f;
    }

    static unsafe int sample_mult(float* probabilities, int n, float coin)
    {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        float cdf = 0.0f;
        for (int i = 0; i < n; i++)
        {
            cdf += probabilities[i];
            if (coin < cdf)
            {
                return i;
            }
        }
        return n - 1; // in case of rounding errors
    }


    static unsafe void ReadExactlyUnmanaged<T>(this FileStream file, Span<T> values)
        where T : unmanaged
    {
        fixed (T* ptr = values)
        {
            ReadExactlyUnmanaged(file, ptr, values.Length);
        }
    }

    static unsafe void ReadExactlyUnmanaged<T>(this FileStream file, T* values, long count)
        where T : unmanaged
    {
        Span<T> buffer = stackalloc T[(256 * 1024) / Unsafe.SizeOf<T>()];
        var totalReadCount = 0;
        while (totalReadCount < count)
        {
            var countToRead = (int)Math.Min(buffer.Length, count - totalReadCount);
            var bufferToRead = buffer.Slice(0, countToRead);
            var span = MemoryMarshal.Cast<T, byte>(bufferToRead);
            file.ReadExactly(span);
            bufferToRead.CopyTo(new Span<T>(values + totalReadCount, countToRead));
            totalReadCount += countToRead;
        }
    }

}
