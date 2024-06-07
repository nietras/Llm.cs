using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

internal static partial class Gpt2
{
    static readonly Action<string> Log = t => { Console.WriteLine(t); Trace.WriteLine(t); };
    static readonly Action<string> LogNoNewLine = t => { Console.Write(t); Trace.Write(t); };

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
    public unsafe static float* AllocateAndPointParameters(ParameterTensors* parameters, long* param_sizes)
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
        &parameters->wte,
            &parameters->wpe,
            &parameters->ln1w,
            &parameters->ln1b,
            &parameters->qkvw,
            &parameters->qkvb,
            &parameters->attprojw,
            &parameters->attprojb,
            &parameters->ln2w,
            &parameters->ln2b,
            &parameters->fcw,
            &parameters->fcb,
            &parameters->fcprojw,
            &parameters->fcprojb,
            &parameters->lnfw,
            &parameters->lnfb
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
        public float* probabilities; // (B, T, V)
        public float* losses; // (B, T)
    }

    public unsafe static float* AllocateAndPointActivations(ActivationTensors* acts, long* act_sizes)
    {
        long num_activations = 0;
        for (long i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        {
            num_activations += act_sizes[i];
        }
        float* acts_memory = malloc<float>(num_activations);
        float**[] ptrs = [
            &acts->encoded,
            &acts->ln1,
            &acts->ln1_mean,
            &acts->ln1_rstd,
            &acts->qkv,
            &acts->atty,
            &acts->preatt,
            &acts->att,
            &acts->attproj,
            &acts->residual2,
            &acts->ln2,
            &acts->ln2_mean,
            &acts->ln2_rstd,
            &acts->fch,
            &acts->fch_gelu,
            &acts->fcproj,
            &acts->residual3,
            &acts->lnf,
            &acts->lnf_mean,
            &acts->lnf_rstd,
            &acts->logits,
            &acts->probabilities,
            &acts->losses
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
        public int* targetTokenIndices; // the target tokens for the current forward pass
        public float mean_loss; // after a forward pass with targetTokenIndices, will be populated with the mean loss
    }

    public unsafe static void BuildFromCheckpoint(GPT2* model, string checkpoint_path)
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
        model->params_memory = AllocateAndPointParameters(&model->parameters, model->param_sizes);
        Extensions.ReadExactlyUnmanaged(model_file, model->params_memory, num_parameters);

        // other inits
        model->acts_memory = null;
        model->grads_memory = null;
        model->m_memory = null;
        model->v_memory = null;
        model->grads_acts_memory = null;
        model->inputs = null;
        model->targetTokenIndices = null;
        model->batch_size = 0;
        model->seq_len = 0;
        model->mean_loss = -1.0f; // -1.0f will designate no loss
    }

    static unsafe void Forward(GPT2* model, int* inputs, int* targetTokenIndices, int B, int T, TimeLlm<Llm> llm)
    {
        // targetTokenIndices are optional and could be null

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
            model->act_sizes[21] = B * T * V; // probabilities
            model->act_sizes[22] = B * T; // losses
            long num_activations = 0;
            for (long i = 0; i < NUM_ACTIVATION_TENSORS; i++)
            {
                num_activations += model->act_sizes[i];
            }
            Log($"num_activations: {num_activations}");
            model->num_activations = num_activations;
            model->acts_memory = AllocateAndPointActivations(&model->acts, model->act_sizes);
            // also create memory for caching inputs and targetTokenIndices
            model->inputs = malloc<int>(B * T);
            model->targetTokenIndices = malloc<int>(B * T); // might be unused if we never have targetTokenIndices but it's small
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

        // cache the inputs/targetTokenIndices
        memcpy(model->inputs, inputs, B * T);
        if (targetTokenIndices != null)
        {
            memcpy(model->targetTokenIndices, targetTokenIndices, B * T);
        }

        llm.Part = "0." + nameof(Forward);
        llm.Index = -1;

        // forward pass
        ParameterTensors parameters = model->parameters; // for brevity
        ActivationTensors acts = model->acts;
        llm.EmbedForward(inputs, parameters.wte, parameters.wpe, B, T, C, acts.encoded); // encoding goes into residual[0]
        var layersStartIndex = llm.Index;
        for (int l = 0; l < L; l++)
        {
            llm.Index = layersStartIndex;
            var residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

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
            llm.LayerNormForward(residual, l_ln1w, l_ln1b, B, T, C, l_ln1_mean, l_ln1_rstd, l_ln1);
            llm.MatMulForward(l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C, l_qkv);
            llm.AttentionForward(l_qkv, B, T, C, NH, l_preatt, l_att, l_atty);
            llm.MatMulForward(l_atty, l_attprojw, l_attprojb, B, T, C, C, l_attproj);
            llm.ResidualForward(residual, l_attproj, B * T * C, l_residual2);
            llm.LayerNormForward(l_residual2, l_ln2w, l_ln2b, B, T, C, l_ln2_mean, l_ln2_rstd, l_ln2);
            llm.MatMulForward(l_ln2, l_fcw, l_fcb, B, T, C, 4 * C, l_fch);
            llm.GeLUForward(l_fch, B * T * 4 * C, l_fch_gelu);
            llm.MatMulForward(l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C, l_fcproj);
            llm.ResidualForward(l_residual2, l_fcproj, B * T * C, l_residual3);
        }
        var lastResidual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3
        llm.LayerNormForward(lastResidual, parameters.lnfw, parameters.lnfb, B, T, C, acts.lnf_mean, acts.lnf_rstd, acts.lnf);
        llm.MatMulForward(acts.lnf, parameters.wte, null, B, T, C, V, acts.logits);
        llm.SoftmaxForward(acts.logits, B, T, V, acts.probabilities);

        // also forward the cross-entropy loss function if we have the targetTokenIndices
        if (targetTokenIndices != null)
        {
            llm.CrossEntropyForward(model->acts.probabilities, targetTokenIndices, B, T, V, model->acts.losses);
            // for convenience also evaluate the mean loss
            float mean_loss = 0.0f;
            for (int i = 0; i < B * T; i++) { mean_loss += model->acts.losses[i]; }
            mean_loss /= B * T;
            model->mean_loss = mean_loss;
        }
        else
        {
            // if we don't have targetTokenIndices, we don't have a loss
            model->mean_loss = -1.0f;
        }
    }

    static unsafe void ZeroGrad(GPT2* model, TimeLlm<Llm> llm)
    {
        llm.Part = "1." + nameof(ZeroGrad);
        llm.Index = -1;
        if (model->grads_memory != null) { llm.Zero(model->grads_memory, model->num_parameters); }
        if (model->grads_acts_memory != null) { llm.Zero(model->grads_acts_memory, model->num_activations); }
    }

    static unsafe void Backward(GPT2* model, TimeLlm<Llm> llm)
    {

        // double check we forwarded previously, with targetTokenIndices
        if (model->mean_loss == -1.0f)
        {
            throw new InvalidOperationException("Error: must forward with targetTokenIndices before backward");
        }

        // lazily allocate the memory for gradients of the weights and activations, if needed
        if (model->grads_memory == null)
        {
            model->grads_memory = AllocateAndPointParameters(&model->grads, model->param_sizes);
            model->grads_acts_memory = AllocateAndPointActivations(&model->grads_acts, model->act_sizes);
            ZeroGrad(model, llm);
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

        llm.Part = "2." + nameof(Backward);
        llm.Index = -1;

        llm.CrossEntropySoftmaxBackward(grads_acts.losses, acts.probabilities, model->targetTokenIndices, B, T, V, grads_acts.logits);
        llm.MatMulBackward(grads_acts.logits, acts.lnf, parameters.wte, B, T, C, V, grads.wte, null, grads_acts.lnf);
        float* residual = acts.residual3 + (L - 1) * B * T * C; // last layer's residual
        float* dresidual = grads_acts.residual3 + (L - 1) * B * T * C; // write to last layer's residual
        llm.LayerNormBackward(grads_acts.lnf, residual, parameters.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C, grads.lnfw, grads.lnfb, dresidual);

        var layerStartIndex = llm.Index;
        for (int l = L - 1; l >= 0; l--)
        {
            llm.Index = layerStartIndex;

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
            llm.ResidualBackward(dl_residual3, B * T * C, dl_residual2, dl_fcproj);
            llm.MatMulBackward(dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C, dl_fcprojw, dl_fcprojb, dl_fch_gelu);
            llm.GeLUBackward(dl_fch_gelu, l_fch, B * T * 4 * C, dl_fch);
            llm.MatMulBackward(dl_fch, l_ln2, l_fcw, B, T, C, 4 * C, dl_fcw, dl_fcb, dl_ln2);
            llm.LayerNormBackward(dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, dl_ln2w, dl_ln2b, dl_residual2);
            llm.ResidualBackward(dl_residual2, B * T * C, dresidual, dl_attproj);
            llm.MatMulBackward(dl_attproj, l_atty, l_attprojw, B, T, C, C, dl_attprojw, dl_attprojb, dl_atty);
            llm.AttentionBackward(dl_atty, l_att, l_qkv, B, T, C, NH, dl_preatt, dl_att, dl_qkv);
            llm.MatMulBackward(dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C, dl_qkvw, dl_qkvb, dl_ln1);
            llm.LayerNormBackward(dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, dl_ln1w, dl_ln1b, dresidual);
        }
        llm.EmbedBackward(grads_acts.encoded, model->inputs, B, T, C, grads.wte, grads.wpe);
    }

    public static unsafe void Update(GPT2* model,
        float learningRate, float beta1, float beta2, float eps, float weightDecay, int t, TimeLlm<Llm> llm)
    {
        // lazily allocate the memory for m_memory and v_memory
        if (model->m_memory == null)
        {
            model->m_memory = calloc<float>(model->num_parameters);
            model->v_memory = calloc<float>(model->num_parameters);
        }
        var parameters = model->params_memory;
        var gradients = model->grads_memory;
        var ms = model->m_memory;
        var vs = model->v_memory;
        var parameterCount = model->num_parameters;

        llm.Part = "3." + nameof(Update);
        llm.Index = -1;

        llm.AdamW(gradients, ms, vs, parameters, parameterCount,
                  learningRate, beta1, beta2, eps, weightDecay, t);
    }

    static unsafe void Free(GPT2* model)
    {
        free(model->params_memory);
        free(model->grads_memory);
        free(model->m_memory);
        free(model->v_memory);
        free(model->acts_memory);
        free(model->grads_acts_memory);
        free(model->inputs);
        free(model->targetTokenIndices);
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
        public int* targetTokenIndices;
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

            // allocate space for B*T + 1 integers to store the inputs and targetTokenIndices
            this.batch = malloc<int>((B * T + 1));
            this.inputs = this.batch;
            this.targetTokenIndices = this.batch + 1; // targetTokenIndices are shifted by one
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


}
