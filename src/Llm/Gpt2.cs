using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

internal static partial class Gpt2
{
    static readonly Action<string> Log = t => { Console.WriteLine(t); Trace.WriteLine(t); };
    static readonly Action<string> LogNoNewLine = t => { Console.Write(t); Trace.Write(t); };

    // Wrap implementation of LLM methods in timing capable
    static TimeLlm CreateTimeLlm(ILlm llm) => new(llm);
    // Skip detailed timing for initial steps
    const int JitAndWarmupCount = 3;

    // ----------------------------------------------------------------------------
    // GPT-2 model definition
    // ----------------------------------------------------------------------------
    // the GPT-2 end-of-text token id
    const int EndOfTextTokenIndex = 50256;

    public sealed class Model(Config config) : IDisposable
    {
        public Config Config { get; } = config;

        // weights (parameters) of the model, and their sizes
        public ParameterTensors Parameters { get; } = ParameterTensors.Create(config);
        // gradients of the weights (parameters)
        public ParameterTensors? ParameterGradients { get; set; } = null;
        // buffers for the AdamW optimizer
        public OptimizerTensors? OptimizerStates { get; set; } = null;

        // outputs of the model, and their sizes
        public OutputTensors? Outputs { get; set; } = null;
        // gradients of the outputs
        public OutputTensors? OutputGradients { get; set; } = null;

        // other run state configuration
        public int Batchsize = 0; // the batch size (B) of current forward pass
        public int TokenCount = 0; // the sequence length (T) of current forward pass

        [MemberNotNull(nameof(Outputs))]
        public void EnsureOutputMemory(int B, int T)
        {
            // allocate space for all the outputs if needed (done here, lazily)
            if (Outputs is null)
            {
                // record the current B,T as well
                Batchsize = B;
                TokenCount = T;
                Outputs = OutputTensors.Create(B, T, Config);
                Log($"OutputCount: {Outputs.TotalCount}");
            }
            else
            {
                // validate B,T is no larger than what was previously allocated
                // in principle, we could re-allocate a larger chunk of memory, for now we just error output
                if (B > Batchsize || T > TokenCount)
                {
                    throw new InvalidDataException("Batch size or token count is inadequately large" +
                        $"Model: B={Batchsize} T={TokenCount}, Desired: B={B} T={T}");
                }
            }
        }

        public void Dispose()
        {
            Parameters.Dispose();
            ParameterGradients?.Dispose();
            OptimizerStates?.Dispose();
            Outputs?.Dispose();
            OutputGradients?.Dispose();
        }
    }

    public record Config
    {
        public int MaxTokenCount; // max sequence length, e.g. 1024
        public int VocabularySize; // vocab size, e.g. 50257
        public int LayerCount; // number of layers, e.g. 12
        public int HeadCount; // number of heads in attention, e.g. 12
        public int ChannelCount; // number of channels, e.g. 768
    }

    public sealed class ParameterTensors(Config c, object s) : Tensors<float>(s)
    {
        public static ParameterTensors Create(Config c)
            => Create<ParameterTensors>(s => new(c, s));

        // Implicitly depends on property initialization following declared
        // order of properties.

        public Tensor<float> TokenEmbeddings { get; } = New([c.VocabularySize, c.ChannelCount], s);
        public Tensor<float> PositionEmbeddings { get; } = New([c.MaxTokenCount, c.ChannelCount], s);

        public Tensor<float> LayerNorm1Weights { get; } = New([c.LayerCount, c.ChannelCount], s);
        public Tensor<float> LayerNorm1Bias { get; } = New([c.LayerCount, c.ChannelCount], s);

        public Tensor<float> QKVWeights { get; } = New([c.LayerCount, 3 * c.ChannelCount, c.ChannelCount], s);
        public Tensor<float> QKVBias { get; } = New([c.LayerCount, 3 * c.ChannelCount], s);

        public Tensor<float> AttentionProjectionWeights { get; } = New([c.LayerCount, c.ChannelCount, c.ChannelCount], s);
        public Tensor<float> AttentionProjectionBias { get; } = New([c.LayerCount, c.ChannelCount], s);

        public Tensor<float> LayerNorm2Weights { get; } = New([c.LayerCount, c.ChannelCount], s);
        public Tensor<float> LayerNorm2Bias { get; } = New([c.LayerCount, c.ChannelCount], s);

        public Tensor<float> FullConnectWeights { get; } = New([c.LayerCount, 4 * c.ChannelCount, c.ChannelCount], s);
        public Tensor<float> FullConnectBias { get; } = New([c.LayerCount, 4 * c.ChannelCount], s);

        public Tensor<float> FullConnectProjectionWeights { get; } = New([c.LayerCount, c.ChannelCount, 4 * c.ChannelCount], s);
        public Tensor<float> FullConnectProjectionBias { get; } = New([c.LayerCount, c.ChannelCount], s);

        public Tensor<float> LayerNormFinalWeights { get; } = New([c.ChannelCount], s);
        public Tensor<float> LayerNormFinalBias { get; } = New([c.ChannelCount], s);
    }

    public sealed class OutputTensors(nint L, nint B, nint T, nint C, nint H, nint V, object s) : Tensors<float>(s)
    {
        public static OutputTensors Create(nint batchSize, nint tokenCount, Config c)
            => Create<OutputTensors>(s => new(c.LayerCount,
                batchSize, tokenCount, c.ChannelCount,
                c.HeadCount, c.VocabularySize, s));

        public Tensor<float> Embeded { get; } = New([B, T, C], s);

        public Tensor<float> LayerNorm1 { get; } = New([L, B, T, C], s);
        public Tensor<float> LayerNorm1Mean { get; } = New([L, B, T], s);
        public Tensor<float> LayerNorm1InvStdDev { get; } = New([L, B, T], s);
        public Tensor<float> QueryKeyValue { get; } = New([L, B, T, 3 * C], s);
        public Tensor<float> Attention { get; } = New([L, B, T, C], s);
        public Tensor<float> PreAttention { get; } = New([L, B, H, T, T], s);
        public Tensor<float> PostAttention { get; } = New([L, B, H, T, T], s);
        public Tensor<float> AttentionProjected { get; } = New([L, B, T, C], s);
        public Tensor<float> Residual2 { get; } = New([L, B, T, C], s);
        public Tensor<float> LayerNorm2 { get; } = New([L, B, T, C], s);
        public Tensor<float> LayerNorm2Mean { get; } = New([L, B, T], s);
        public Tensor<float> LayerNorm2InvStdDev { get; } = New([L, B, T], s);
        public Tensor<float> FullyConnected { get; } = New([L, B, T, 4 * C], s);
        public Tensor<float> FullyConnectedGeLU { get; } = New([L, B, T, 4 * C], s);
        public Tensor<float> FullyConnectedProjected { get; } = New([L, B, T, C], s);
        public Tensor<float> Residual3 { get; } = New([L, B, T, C], s);

        public Tensor<float> LayerNormFinal { get; } = New([B, T, C], s);
        public Tensor<float> LayerNormFinalMean { get; } = New([B, T], s);
        public Tensor<float> LayerNormFinalInvStdDev { get; } = New([B, T], s);
        public Tensor<float> Logits { get; } = New([B, T, V], s);
        public Tensor<float> Probabilities { get; } = New([B, T, V], s);
        public Tensor<float> Losses { get; } = New([B, T], s);
    }

    public sealed class OptimizerTensors(nint parameterCount, object s) : Tensors<float>(s)
    {
        public static OptimizerTensors Create(nint parameterCount)
            => Create<OptimizerTensors>(s => new(parameterCount, s));

        public Tensor<float> M { get; } = New([parameterCount], s);
        public Tensor<float> V { get; } = New([parameterCount], s);
    }

    public unsafe static Model ModelFromCheckpoint(string checkpointFilePath)
    {
        // read in model from a checkpoint file
        using var file = File.OpenRead(checkpointFilePath);
        Span<int> header = stackalloc int[256];
        // read span from model_file
        file.ReadExactlyUnmanaged(header);
        //fread(model_header, sizeof(int), 256, model_file);
        if (header[0] != 20240326) { throw new InvalidDataException($"Bad magic model file"); }
        if (header[1] != 1) { throw new InvalidDataException($"Bad version in model file"); }

        // read in hyperparameters
        var config = new Config()
        {
            MaxTokenCount = header[2],
            VocabularySize = header[3],
            LayerCount = header[4],
            HeadCount = header[5],
            ChannelCount = header[6],
        };
        Log("[GPT-2]");
        Log($"MaxTokenCount: {config.MaxTokenCount}");
        Log($"VocabularySize: {config.VocabularySize}");
        Log($"LayerCount: {config.LayerCount}");
        Log($"HeadCount: {config.HeadCount}");
        Log($"ChannelCount: {config.ChannelCount}");

        var model = new Model(config);
        // read in all the parameters from file
        file.ReadExactlyUnmanaged(model.Parameters.MemoryPtr, model.Parameters.TotalCount);
        Log($"ParameterCount: {model.Parameters.TotalCount}");

        return model;
    }

    internal readonly record struct TrainStepTimings(double Total_ms,
        double Forward_ms, double ZeroGrad_ms, double Backward_ms, double Update_ms);
    internal readonly record struct TrainStepResult(float Loss, TrainStepTimings Timings);

    static readonly double s_tickstoMs = 1000.0 / Stopwatch.Frequency;

    internal static unsafe string ToReport(this TrainStepTimings t)
    {
        return $"{t.Total_ms,5:F0} ms = Forward {t.Forward_ms,5:F0} ms ZeroGrad {t.ZeroGrad_ms,3:F0} ms " +
               $"Backward {t.Backward_ms,4:F0} ms Update {t.Update_ms,4:F0} ms";
    }

    internal static unsafe TrainStepResult TrainStep(Model model,
        int* inputTokenIndices, int* targetTokenIndices, int batchSize, int tokenCount,
        TimeLlm llm, int step)
    {
        var t0 = Stopwatch.GetTimestamp();
        var loss = Forward(model, inputTokenIndices, targetTokenIndices,
                           batchSize, tokenCount, llm);
        var t1 = Stopwatch.GetTimestamp();
        ZeroGrad(model, llm);
        var t2 = Stopwatch.GetTimestamp();
        Backward(model, inputTokenIndices, targetTokenIndices, llm);
        var t3 = Stopwatch.GetTimestamp();
        Update(model, learningRate: 1e-4f, beta1: 0.9f, beta2: 0.999f,
               eps: 1e-8f, weightDecay: 0.01f, step + 1, llm);
        var t4 = Stopwatch.GetTimestamp();

        TrainStepTimings timings = new((t4 - t0) * s_tickstoMs,
            (t1 - t0) * s_tickstoMs, (t2 - t1) * s_tickstoMs,
            (t3 - t2) * s_tickstoMs, (t4 - t3) * s_tickstoMs);
        return new(loss, timings);
    }

    static unsafe float Forward(Model model, int* inputs,
        int* targetTokenIndices, int B, int T, TimeLlm llm)
    {
        // targetTokenIndices are optional and could be null

        // ensure the model was initialized or error output
        if (model.Parameters.MemoryPtr == null)
        {
            throw new InvalidOperationException("Model was not initialized properly.");
        }

        // convenience parameters
        int V = model.Config.VocabularySize;
        int L = model.Config.LayerCount;
        int H = model.Config.HeadCount;
        int C = model.Config.ChannelCount;

        model.EnsureOutputMemory(B, T);

        llm.Part = "0." + nameof(Forward);
        llm.Index = -1;

        // forward pass
        var parameters = model.Parameters; // for brevity
        var outputs = model.Outputs;

        llm.EmbedForward(inputs, parameters.TokenEmbeddings, parameters.PositionEmbeddings, B, T, C, outputs.Embeded);
        var layersStartIndex = llm.Index;
        for (int l = 0; l < L; l++)
        {
            llm.Index = layersStartIndex;
            var residual = l == 0 ? outputs.Embeded : outputs.Residual3.StrideToPtrAt(l - 1);

            // get the pointers of the weights for this layer
            float* l_ln1w = parameters.LayerNorm1Weights.StrideToPtrAt(l);
            float* l_ln1b = parameters.LayerNorm1Bias.StrideToPtrAt(l);
            float* l_qkvw = parameters.QKVWeights.StrideToPtrAt(l);
            float* l_qkvb = parameters.QKVBias.StrideToPtrAt(l);
            float* l_attprojw = parameters.AttentionProjectionWeights.StrideToPtrAt(l);
            float* l_attprojb = parameters.AttentionProjectionBias.StrideToPtrAt(l);
            float* l_ln2w = parameters.LayerNorm2Weights.StrideToPtrAt(l);
            float* l_ln2b = parameters.LayerNorm2Bias.StrideToPtrAt(l);
            float* l_fcw = parameters.FullConnectWeights.StrideToPtrAt(l);
            float* l_fcb = parameters.FullConnectBias.StrideToPtrAt(l);
            float* l_fcprojw = parameters.FullConnectProjectionWeights.StrideToPtrAt(l);
            float* l_fcprojb = parameters.FullConnectProjectionBias.StrideToPtrAt(l);

            // get the pointers of the outputs for this layer
            float* l_ln1 = outputs.LayerNorm1.StrideToPtrAt(l);
            float* l_ln1_mean = outputs.LayerNorm1Mean.StrideToPtrAt(l);
            float* l_ln1_rstd = outputs.LayerNorm1InvStdDev.StrideToPtrAt(l);
            float* l_qkv = outputs.QueryKeyValue.StrideToPtrAt(l);
            float* l_atty = outputs.Attention.StrideToPtrAt(l);
            float* l_preatt = outputs.PreAttention.StrideToPtrAt(l);
            float* l_att = outputs.PostAttention.StrideToPtrAt(l);
            float* l_attproj = outputs.AttentionProjected.StrideToPtrAt(l);
            float* l_residual2 = outputs.Residual2.StrideToPtrAt(l);
            float* l_ln2 = outputs.LayerNorm2.StrideToPtrAt(l);
            float* l_ln2_mean = outputs.LayerNorm2Mean.StrideToPtrAt(l);
            float* l_ln2_rstd = outputs.LayerNorm2InvStdDev.StrideToPtrAt(l);
            float* l_fch = outputs.FullyConnected.StrideToPtrAt(l);
            float* l_fch_gelu = outputs.FullyConnectedGeLU.StrideToPtrAt(l);
            float* l_fcproj = outputs.FullyConnectedProjected.StrideToPtrAt(l);
            float* l_residual3 = outputs.Residual3.StrideToPtrAt(l);

            // now do the forward pass
            llm.LayerNormForward(residual, l_ln1w, l_ln1b, B, T, C, l_ln1_mean, l_ln1_rstd, l_ln1);
            llm.MatMulForward(l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C, l_qkv);
            llm.AttentionForward(l_qkv, B, T, C, H, l_preatt, l_att, l_atty);
            llm.MatMulForward(l_atty, l_attprojw, l_attprojb, B, T, C, C, l_attproj);
            llm.ResidualForward(residual, l_attproj, B * T * C, l_residual2);
            llm.LayerNormForward(l_residual2, l_ln2w, l_ln2b, B, T, C, l_ln2_mean, l_ln2_rstd, l_ln2);
            llm.MatMulForward(l_ln2, l_fcw, l_fcb, B, T, C, 4 * C, l_fch);
            llm.GeLUForward(l_fch, B * T * 4 * C, l_fch_gelu);
            llm.MatMulForward(l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C, l_fcproj);
            llm.ResidualForward(l_residual2, l_fcproj, B * T * C, l_residual3);
        }
        // last residual is in residual3
        var lastResidual = outputs.Residual3.StrideToPtrAt(L - 1);
        llm.LayerNormForward(lastResidual, parameters.LayerNormFinalWeights, parameters.LayerNormFinalBias, B, T, C,
                             outputs.LayerNormFinalMean, outputs.LayerNormFinalInvStdDev, outputs.LayerNormFinal);
        llm.MatMulForward(outputs.LayerNormFinal, parameters.TokenEmbeddings, null, B, T, C, V, outputs.Logits);
        llm.SoftmaxForward(outputs.Logits, B, T, V, outputs.Probabilities);

        // also forward the cross-entropy loss function if we have the targetTokenIndices
        if (targetTokenIndices != null)
        {
            llm.CrossEntropyForward(model.Outputs.Probabilities, targetTokenIndices, B, T, V, model.Outputs.Losses);
            // for convenience also evaluate the mean loss
            float meanLoss = 0.0f;
            for (int i = 0; i < B * T; i++) { meanLoss += model.Outputs.Losses.Ptr[i]; }
            meanLoss /= B * T;
            return meanLoss;
        }
        else
        {
            // if we don't have targetTokenIndices, we don't have a loss
            return -1;
        }

    }

    static unsafe void ZeroGrad(Model model, TimeLlm llm)
    {
        llm.Part = "1." + nameof(ZeroGrad);
        llm.Index = -1;
        // lazily allocate the memory for gradients of the weights and outputs
        model.ParameterGradients ??= ParameterTensors.Create(model.Config);
        model.OutputGradients ??= OutputTensors.Create(model.Batchsize, model.TokenCount, model.Config);
        llm.Zero(model.ParameterGradients.MemoryPtr, model.ParameterGradients.TotalCount);
        llm.Zero(model.OutputGradients.MemoryPtr, model.OutputGradients.TotalCount);
    }

    static unsafe void Backward(Model model, int* inputTokenIndices, int* targetTokenIndices, TimeLlm llm)
    {
        // convenience shortcuts
        int B = model.Batchsize;
        int T = model.TokenCount;
        int V = model.Config.VocabularySize;
        int L = model.Config.LayerCount;
        int H = model.Config.HeadCount;
        int C = model.Config.ChannelCount;

        // backward pass: go in the reverse order of the forward pass, and call backward() functions
        var parameters = model.Parameters; // for brevity
        var parameterGradients = model.ParameterGradients!;
        OutputTensors outputs = model.Outputs!;
        OutputTensors outputGradients = model.OutputGradients!;

        // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
        // technically this is a small, inline backward() pass of calculating
        // total, final loss as the mean over all losses over all (B,T) positions in the batch
        float δlossMean = 1.0f / (B * T);
        for (int i = 0; i < B * T; i++) { outputGradients.Losses.Ptr[i] = δlossMean; }

        llm.Part = "2." + nameof(Backward);
        llm.Index = -1;

        llm.CrossEntropySoftmaxBackward(outputGradients.Losses, outputs.Probabilities, targetTokenIndices, B, T, V,
                                        outputGradients.Logits);
        llm.MatMulBackward(outputGradients.Logits, outputs.LayerNormFinal, parameters.TokenEmbeddings, B, T, C, V,
                           parameterGradients.TokenEmbeddings, null, outputGradients.LayerNormFinal);
        float* residual = outputs.Residual3.StrideToPtrAt(L - 1); // last layer's residual
        float* δresidual = outputGradients.Residual3.StrideToPtrAt(L - 1); // write to last layer's residual
        llm.LayerNormBackward(outputGradients.LayerNormFinal, residual,
            parameters.LayerNormFinalWeights, outputs.LayerNormFinalMean, outputs.LayerNormFinalInvStdDev, B, T, C,
            parameterGradients.LayerNormFinalWeights, parameterGradients.LayerNormFinalBias, δresidual);

        var layerStartIndex = llm.Index;
        for (int l = L - 1; l >= 0; l--)
        {
            llm.Index = layerStartIndex;

            residual = l == 0 ? outputs.Embeded : outputs.Residual3.StrideToPtrAt(l - 1);
            δresidual = l == 0 ? outputGradients.Embeded : outputGradients.Residual3.StrideToPtrAt(l - 1);

            // get the pointers of the weights for this layer
            float* l_ln1w = parameters.LayerNorm1Weights.StrideToPtrAt(l);
            float* l_qkvw = parameters.QKVWeights.StrideToPtrAt(l);
            float* l_attprojw = parameters.AttentionProjectionWeights.StrideToPtrAt(l);
            float* l_ln2w = parameters.LayerNorm2Weights.StrideToPtrAt(l);
            float* l_fcw = parameters.FullConnectWeights.StrideToPtrAt(l);
            float* l_fcprojw = parameters.FullConnectProjectionWeights.StrideToPtrAt(l);
            // get the pointers of the gradients of the weights for this layer
            float* dl_ln1w = parameterGradients.LayerNorm1Weights.StrideToPtrAt(l);
            float* dl_ln1b = parameterGradients.LayerNorm1Bias.StrideToPtrAt(l);
            float* dl_qkvw = parameterGradients.QKVWeights.StrideToPtrAt(l);
            float* dl_qkvb = parameterGradients.QKVBias.StrideToPtrAt(l);
            float* dl_attprojw = parameterGradients.AttentionProjectionWeights.StrideToPtrAt(l);
            float* dl_attprojb = parameterGradients.AttentionProjectionBias.StrideToPtrAt(l);
            float* dl_ln2w = parameterGradients.LayerNorm2Weights.StrideToPtrAt(l);
            float* dl_ln2b = parameterGradients.LayerNorm2Bias.StrideToPtrAt(l);
            float* dl_fcw = parameterGradients.FullConnectWeights.StrideToPtrAt(l);
            float* dl_fcb = parameterGradients.FullConnectBias.StrideToPtrAt(l);
            float* dl_fcprojw = parameterGradients.FullConnectProjectionWeights.StrideToPtrAt(l);
            float* dl_fcprojb = parameterGradients.FullConnectProjectionBias.StrideToPtrAt(l);
            // get the pointers of the outputs for this layer
            float* l_ln1 = outputs.LayerNorm1.StrideToPtrAt(l);
            float* l_ln1_mean = outputs.LayerNorm1Mean.StrideToPtrAt(l);
            float* l_ln1_rstd = outputs.LayerNorm1InvStdDev.StrideToPtrAt(l);
            float* l_qkv = outputs.QueryKeyValue.StrideToPtrAt(l);
            float* l_atty = outputs.Attention.StrideToPtrAt(l);
            float* l_att = outputs.PostAttention.StrideToPtrAt(l);
            float* l_residual2 = outputs.Residual2.StrideToPtrAt(l);
            float* l_ln2 = outputs.LayerNorm2.StrideToPtrAt(l);
            float* l_ln2_mean = outputs.LayerNorm2Mean.StrideToPtrAt(l);
            float* l_ln2_rstd = outputs.LayerNorm2InvStdDev.StrideToPtrAt(l);
            float* l_fch = outputs.FullyConnected.StrideToPtrAt(l);
            float* l_fch_gelu = outputs.FullyConnectedGeLU.StrideToPtrAt(l);
            // get the pointers of the gradients of the outputs for this layer
            float* dl_ln1 = outputGradients.LayerNorm1.StrideToPtrAt(l);
            float* dl_qkv = outputGradients.QueryKeyValue.StrideToPtrAt(l);
            float* dl_atty = outputGradients.Attention.StrideToPtrAt(l);
            float* dl_preatt = outputGradients.PreAttention.StrideToPtrAt(l);
            float* dl_att = outputGradients.PostAttention.StrideToPtrAt(l);
            float* dl_attproj = outputGradients.AttentionProjected.StrideToPtrAt(l);
            float* dl_residual2 = outputGradients.Residual2.StrideToPtrAt(l);
            float* dl_ln2 = outputGradients.LayerNorm2.StrideToPtrAt(l);
            float* dl_fch = outputGradients.FullyConnected.StrideToPtrAt(l);
            float* dl_fch_gelu = outputGradients.FullyConnectedGeLU.StrideToPtrAt(l);
            float* dl_fcproj = outputGradients.FullyConnectedProjected.StrideToPtrAt(l);
            float* dl_residual3 = outputGradients.Residual3.StrideToPtrAt(l);

            // backprop this layer
            llm.ResidualBackward(dl_residual3, B * T * C, dl_residual2, dl_fcproj);
            llm.MatMulBackward(dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C, dl_fcprojw, dl_fcprojb, dl_fch_gelu);
            llm.GeLUBackward(dl_fch_gelu, l_fch, B * T * 4 * C, dl_fch);
            llm.MatMulBackward(dl_fch, l_ln2, l_fcw, B, T, C, 4 * C, dl_fcw, dl_fcb, dl_ln2);
            llm.LayerNormBackward(dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, dl_ln2w, dl_ln2b, dl_residual2);
            llm.ResidualBackward(dl_residual2, B * T * C, δresidual, dl_attproj);
            llm.MatMulBackward(dl_attproj, l_atty, l_attprojw, B, T, C, C, dl_attprojw, dl_attprojb, dl_atty);
            llm.AttentionBackward(dl_atty, l_att, l_qkv, B, T, C, H, dl_preatt, dl_att, dl_qkv);
            llm.MatMulBackward(dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C, dl_qkvw, dl_qkvb, dl_ln1);
            llm.LayerNormBackward(dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, dl_ln1w, dl_ln1b, δresidual);
        }
        llm.EmbedBackward(outputGradients.Embeded, inputTokenIndices, B, T, C,
            parameterGradients.TokenEmbeddings, parameterGradients.PositionEmbeddings);
    }

    public static unsafe void Update(Model model,
        float learningRate, float beta1, float beta2,
        float eps, float weightDecay, int t, TimeLlm llm)
    {
        llm.Part = "3." + nameof(Update);
        llm.Index = -1;

        // lazily allocate the memory for optimizer
        if (model.OptimizerStates is null)
        {
            model.OptimizerStates = OptimizerTensors.Create(model.Parameters.TotalCount);
            llm.Zero(model.OptimizerStates.MemoryPtr, model.OptimizerStates.TotalCount);
        }
        var parameters = model.Parameters.MemoryPtr;
        var gradients = model.ParameterGradients!.MemoryPtr;
        var ms = model.OptimizerStates.M.Ptr;
        var vs = model.OptimizerStates.V.Ptr;
        var parameterCount = model.Parameters.TotalCount;

        llm.AdamW(gradients, ms, vs, parameters, parameterCount,
                  learningRate, beta1, beta2, eps, weightDecay, t);
    }

    // ----------------------------------------------------------------------------
    // data loader lite
    // returns random batches of data from a file of integers

    public unsafe class DataLoader : IDisposable
    {
        // input handling and its state
        FileStream _tokensFile;
        readonly long _fileSize;
        Ntv<int> _batchMemory;

        // hyperparameters
        public readonly int BatchSize;
        public readonly int TokenCount;
        // output pointers
        public int* BatchTokenIndices;
        public int* InputTokenIndices;
        public int* TargetTokenIndices;
        // convenience variables
        public nint BatchCount;

        public DataLoader(string filename, int B, int T)
        {
            BatchSize = B;
            TokenCount = T;
            var batchTokenCount = B * T + 1;

            // open the input file for reading
            _tokensFile = File.OpenRead(filename);
            _fileSize = _tokensFile.Length;
            // allocate space for B*T + 1 integers to store the inputTokenIndices and targetTokenIndices
            _batchMemory = new Ntv<int>(batchTokenCount);
            if (_fileSize < _batchMemory.ByteCount)
            {
                throw new InvalidDataException($"File size is too small for the batch size and token count");
            }

            BatchTokenIndices = _batchMemory.Ptr;
            InputTokenIndices = BatchTokenIndices;
            TargetTokenIndices = BatchTokenIndices + 1; // targetTokenIndices are shifted by one
            BatchCount = (nint)((_fileSize - 1) / (B * T * sizeof(int)));
        }

        public unsafe void Reset()
        {
            _tokensFile.Position = 0;
        }

        public unsafe void NextBatch()
        {
            // If at the end of the file, loop back to the beginning
            if (_tokensFile.Position + _batchMemory.ByteCount > _fileSize)
            {
                _tokensFile.Position = 0;
            }
            // Read the B*T+1 integers from the file into batch
            _tokensFile.ReadExactlyUnmanaged(_batchMemory.Ptr, _batchMemory.Count);
            // Read +1 more token to get the target and hence have to move back
            _tokensFile.Position -= sizeof(int);
        }

        public void DisposeManaged()
        {
            _tokensFile.Dispose();
            _tokensFile = null!;
            _batchMemory.Dispose();
            _batchMemory = null!;
            BatchTokenIndices = null;
            InputTokenIndices = null;
            TargetTokenIndices = null;
        }

        #region Dispose
        protected virtual void Dispose(bool disposing)
        {
            if (!_disposedValue)
            {
                if (disposing)
                {
                    DisposeManaged();
                }
                _disposedValue = true;
            }
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        bool _disposedValue;
        #endregion
    }

    // ----------------------------------------------------------------------------
    // sampler

    static unsafe float RandomSingle(ulong* state)
    { // random float32 in [0,1)
        return (RandomUInt32(state) >> 8) / 16777216.0f;
    }
    static unsafe uint RandomUInt32(ulong* state)
    {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (uint)((*state * 0x2545F4914F6CDD1Dul) >> 32);
    }

    static unsafe int FindSampleIndex(float* probabilities, int n, float coin)
    {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from RandomSingle()
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
