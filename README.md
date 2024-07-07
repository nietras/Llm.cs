# Llm.cs - C# port of @karpathy [llm.c](https://github.com/karpathy/llm.c)
![.NET](https://img.shields.io/badge/net8.0-5C2D91?logo=.NET&labelColor=gray)
![C#](https://img.shields.io/badge/12.0-239120?logo=csharp&logoColor=white&labelColor=gray)
![Lines of code](https://tokei.rs/b1/github/nietras/Llm.cs?category=code)
[![Build Status](https://github.com/nietras/Llm.cs/actions/workflows/dotnet.yml/badge.svg?branch=main)](https://github.com/nietras/Llm.cs/actions/workflows/dotnet.yml)
[![Super-Linter](https://github.com/nietras/Llm.cs/actions/workflows/super-linter.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![codecov](https://codecov.io/gh/nietras/Llm.cs/branch/main/graph/badge.svg?token=WN56CR3X0D)](https://codecov.io/gh/nietras/Llm.cs)
[![CodeQL](https://github.com/nietras/Llm.cs/workflows/CodeQL/badge.svg)](https://github.com/nietras/Llm.cs/actions?query=workflow%3ACodeQL)
[![Nuget](https://img.shields.io/nuget/v/Llm?color=purple)](https://www.nuget.org/packages/Llm/)
[![Release](https://img.shields.io/github/v/release/nietras/Llm.cs)](https://github.com/nietras/Llm.cs/releases/)
[![downloads](https://img.shields.io/nuget/dt/Llm)](https://www.nuget.org/packages/Llm)
![Size](https://img.shields.io/github/repo-size/nietras/Llm.cs.svg)
[![License](https://img.shields.io/github/license/nietras/Llm.cs)](https://github.com/nietras/Llm.cs/blob/main/LICENSE)
[![Blog](https://img.shields.io/badge/blog-nietras.com-4993DD)](https://nietras.com)

⭐ Please star this project if you like it. ⭐

## Getting Started

* Install [.NET SDK](https://dotnet.microsoft.com/en-us/download) matching
  version in [global.json](./global.json)
* Restore and build the project (working directory where `Llm.sln`):
  ```powershell
  dotnet restore
  dotnet build -c Release
  ```
* Run (currently defaults to running train "test"/verification):
  ```powershell
  dotnet run -c Release --project .\src\Llm\Llm.csproj
  ```
  NOTE: First time this is run it will download binary files e.g. weights and
  input tokens from
  [nietras/llm.bin](https://huggingface.co/datasets/nietras/llm.bin) on Hugging
  Face. This means there is no need to run any Python here to get data or
  similar. Clone and run ✅
* Output should then be something like:
  ```
  ProcessorCount: 32
  [GPT-2]
  MaxTokenCount: 1024
  VocabularySize: 50257
  LayerCount: 12
  HeadCount: 12
  ChannelCount: 768
  ParameterCount: 124439808
  [State]
  BatchSize: 4
  TokenCount: 64
  OutputCount: 73323776
  Logits                         TENSOR OK   MaxAbsDiff 0.000534
  δTokenEmbeddings               TENSOR OK   MaxAbsDiff 0.001185
  δPositionEmbeddings            TENSOR OK   MaxAbsDiff 0.000037
  δLayerNorm1Weights             TENSOR OK   MaxAbsDiff 0.003039
  δLayerNorm1Bias                TENSOR OK   MaxAbsDiff 0.001283
  δQKVWeights                    TENSOR OK   MaxAbsDiff 0.000474
  δQKVBias                       TENSOR OK   MaxAbsDiff 0.000257
  δAttentionProjectionWeights    TENSOR OK   MaxAbsDiff 0.000200
  δAttentionProjectionBias       TENSOR OK   MaxAbsDiff 0.000179
  δLayerNorm2Weights             TENSOR OK   MaxAbsDiff 0.009708
  δLayerNorm2Bias                TENSOR OK   MaxAbsDiff 0.000819
  δFullConnectWeights            TENSOR OK   MaxAbsDiff 0.000794
  δFullConnectBias               TENSOR OK   MaxAbsDiff 0.000193
  δFullConnectProjectionWeights  TENSOR OK   MaxAbsDiff 0.000385
  δFullConnectProjectionBias     TENSOR OK   MaxAbsDiff 0.000118
  δLayerNormFinalWeights         TENSOR OK   MaxAbsDiff 0.000362
  δLayerNormFinalBias            TENSOR OK   MaxAbsDiff 0.000066
   0: loss 5.269892 exp. 5.270007 OK   ( 1386 ms = Forward   490 ms ZeroGrad  90 ms Backward  632 ms Update  174 ms) JIT/WARMUP
   1: loss 4.059388 exp. 4.059707 OK   (  875 ms = Forward   279 ms ZeroGrad  28 ms Backward  463 ms Update  106 ms) JIT/WARMUP
   2: loss 3.374209 exp. 3.375123 OK   ( 1005 ms = Forward   407 ms ZeroGrad  28 ms Backward  459 ms Update  110 ms) JIT/WARMUP
   3: loss 2.800130 exp. 2.800783 OK   (  867 ms = Forward   266 ms ZeroGrad  28 ms Backward  474 ms Update   99 ms)
   4: loss 2.315308 exp. 2.315382 OK   (  847 ms = Forward   238 ms ZeroGrad  28 ms Backward  477 ms Update  103 ms)
   5: loss 1.849346 exp. 1.849029 OK   (  884 ms = Forward   234 ms ZeroGrad  28 ms Backward  516 ms Update  106 ms)
   6: loss 1.395217 exp. 1.394656 OK   (  884 ms = Forward   282 ms ZeroGrad  28 ms Backward  468 ms Update  106 ms)
   7: loss 0.998617 exp. 0.999147 OK   (  839 ms = Forward   231 ms ZeroGrad  28 ms Backward  474 ms Update  106 ms)
   8: loss 0.625541 exp. 0.624080 OK   (  887 ms = Forward   309 ms ZeroGrad  28 ms Backward  449 ms Update  102 ms)
   9: loss 0.378010 exp. 0.376511 OK   (  915 ms = Forward   311 ms ZeroGrad  28 ms Backward  485 ms Update   91 ms)
  All okay: True
  
  0.Forward  00 EmbedForward                 0% count:   7 sum:    1.2 min:   0.2 mean:   0.2 max:   0.2 [ms]
  0.Forward  01 LayerNormForward             0% count:  84 sum:    7.2 min:   0.1 mean:   0.1 max:   0.1 [ms]
  0.Forward  02 MatMulForward                3% count:  84 sum:  197.5 min:   1.9 mean:   2.4 max:   4.4 [ms]
  0.Forward  03 AttentionForward             1% count:  84 sum:   67.0 min:   0.7 mean:   0.8 max:   1.6 [ms]
  0.Forward  04 MatMulForward                1% count:  84 sum:   69.5 min:   0.7 mean:   0.8 max:   1.4 [ms]
  0.Forward  05 ResidualForward              0% count:  84 sum:    7.5 min:   0.1 mean:   0.1 max:   0.3 [ms]
  0.Forward  06 LayerNormForward             0% count:  84 sum:    6.8 min:   0.1 mean:   0.1 max:   0.1 [ms]
  0.Forward  07 MatMulForward                4% count:  84 sum:  252.7 min:   2.3 mean:   3.0 max:   5.2 [ms]
  0.Forward  08 GeLUForward                  1% count:  84 sum:   74.9 min:   0.7 mean:   0.9 max:   1.7 [ms]
  0.Forward  09 MatMulForward                4% count:  84 sum:  250.0 min:   2.4 mean:   3.0 max:   5.9 [ms]
  0.Forward  10 ResidualForward              0% count:  84 sum:    7.2 min:   0.1 mean:   0.1 max:   0.1 [ms]
  0.Forward  11 LayerNormForward             0% count:   7 sum:    0.5 min:   0.1 mean:   0.1 max:   0.1 [ms]
  0.Forward  12 MatMulForward               15% count:   7 sum:  887.3 min:  93.7 mean: 126.8 max: 175.8 [ms]
  0.Forward  13 SoftmaxForward               1% count:   7 sum:   38.6 min:   3.8 mean:   5.5 max:   9.4 [ms]
  0.Forward  14 CrossEntropyForward          0% count:   7 sum:    0.2 min:   0.0 mean:   0.0 max:   0.0 [ms]
  
  1.ZeroGrad 00 Zero                         2% count:   7 sum:  123.0 min:  17.5 mean:  17.6 max:  17.6 [ms]
  1.ZeroGrad 01 Zero                         1% count:   7 sum:   72.2 min:  10.3 mean:  10.3 max:  10.4 [ms]
  
  2.Backward 00 CrossEntropySoftmaxBackward  1% count:   7 sum:   45.4 min:   6.4 mean:   6.5 max:   6.6 [ms]
  2.Backward 01 MatMulBackward              18% count:   7 sum: 1116.7 min: 139.0 mean: 159.5 max: 173.6 [ms]
  2.Backward 02 LayerNormBackward            0% count:   7 sum:    4.1 min:   0.5 mean:   0.6 max:   0.6 [ms]
  2.Backward 03 ResidualBackward             0% count:  84 sum:   14.1 min:   0.1 mean:   0.2 max:   0.3 [ms]
  2.Backward 04 MatMulBackward              11% count:  84 sum:  679.4 min:   6.8 mean:   8.1 max:  14.3 [ms]
  2.Backward 05 GeLUBackward                 2% count:  84 sum:  124.7 min:   1.3 mean:   1.5 max:   2.0 [ms]
  2.Backward 06 MatMulBackward               8% count:  84 sum:  519.8 min:   5.7 mean:   6.2 max:  10.2 [ms]
  2.Backward 07 LayerNormBackward            1% count:  84 sum:   39.3 min:   0.4 mean:   0.5 max:   1.1 [ms]
  2.Backward 08 ResidualBackward             0% count:  84 sum:   12.2 min:   0.1 mean:   0.1 max:   0.3 [ms]
  2.Backward 09 MatMulBackward               3% count:  84 sum:  158.9 min:   1.6 mean:   1.9 max:   7.1 [ms]
  2.Backward 10 AttentionBackward            3% count:  84 sum:  159.1 min:   1.4 mean:   1.9 max:   4.6 [ms]
  2.Backward 11 MatMulBackward               7% count:  84 sum:  423.6 min:   4.3 mean:   5.0 max:   8.1 [ms]
  2.Backward 12 LayerNormBackward            1% count:  84 sum:   42.5 min:   0.4 mean:   0.5 max:   0.8 [ms]
  2.Backward 13 EmbedBackward                0% count:   7 sum:    1.3 min:   0.2 mean:   0.2 max:   0.2 [ms]
  
  3.Update   00 AdamW                       12% count:   7 sum:  713.1 min:  91.1 mean: 101.9 max: 106.3 [ms]
  
  MatMulBackward               47% sum:   2898 [ms] per step:    414 [ms]
  MatMulForward                27% sum:   1657 [ms] per step:    237 [ms]
  AdamW                        12% sum:    713 [ms] per step:    102 [ms]
  Zero                          3% sum:    195 [ms] per step:     28 [ms]
  AttentionBackward             3% sum:    159 [ms] per step:     23 [ms]
  GeLUBackward                  2% sum:    125 [ms] per step:     18 [ms]
  LayerNormBackward             1% sum:     86 [ms] per step:     12 [ms]
  GeLUForward                   1% sum:     75 [ms] per step:     11 [ms]
  AttentionForward              1% sum:     67 [ms] per step:     10 [ms]
  CrossEntropySoftmaxBackward   1% sum:     45 [ms] per step:      6 [ms]
  SoftmaxForward                1% sum:     39 [ms] per step:      6 [ms]
  ResidualBackward              0% sum:     26 [ms] per step:      4 [ms]
  ResidualForward               0% sum:     15 [ms] per step:      2 [ms]
  LayerNormForward              0% sum:     14 [ms] per step:      2 [ms]
  EmbedBackward                 0% sum:      1 [ms] per step:      0 [ms]
  EmbedForward                  0% sum:      1 [ms] per step:      0 [ms]
  CrossEntropyForward           0% sum:      0 [ms] per step:      0 [ms]
  Total                       100% sum:   6117 [ms] per step:    874 [ms]  
  ```

## Example
```csharp

```

## Public API Reference
```csharp
[assembly: System.CLSCompliant(false)]
[assembly: System.Reflection.AssemblyMetadata("IsTrimmable", "True")]
[assembly: System.Reflection.AssemblyMetadata("RepositoryUrl", "https://github.com/nietras/Llm/")]
[assembly: System.Resources.NeutralResourcesLanguage("en")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Llm.Benchmarks")]
[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Llm.Test")]
[assembly: System.Runtime.Versioning.TargetFramework(".NETCoreApp,Version=v8.0", FrameworkDisplayName=".NET 8.0")]
namespace nietras.LargeLanguageModel
{
    public interface ILlm
    {
        unsafe void AdamW(float* gradients, float* ms, float* vs, float* parameters, System.IntPtr parameterCount, float learningRate, float beta1, float beta2, float eps, float weightDecay, int t);
        unsafe void AttentionBackward(float* δoutput, float* postAttention, float* input, int batchSize, int tokenCount, int channelCount, int headCount, float* δpreAttention, float* δpostAttention, float* δinput);
        unsafe void AttentionForward(float* input, int batchSize, int tokenCount, int channelCount, int headCount, float* preAttention, float* postAttention, float* output);
        unsafe void CrossEntropyForward(float* probabilities, int* targetTokenIndices, int batchSize, int tokenCount, int vocabularySize, float* losses);
        unsafe void CrossEntropySoftmaxBackward(float* δlosses, float* probabilities, int* targetTokenIndices, int batchSize, int tokenCount, int vocabularySize, float* δlogits);
        unsafe void EmbedBackward(float* δoutput, int* tokenIndices, int batchSize, int tokenCount, int channelCount, float* δtokenEmbeddings, float* δpositionEmbeddings);
        unsafe void EmbedForward(int* tokenIndices, float* tokenEmbeddings, float* positionEmbeddings, int batchSize, int tokenCount, int channelCount, float* output);
        unsafe void GeLUBackward(float* δoutput, float* input, int count, float* δinput);
        unsafe void GeLUForward(float* input, int count, float* output);
        unsafe void LayerNormBackward(float* δoutput, float* input, float* weight, float* mean, float* invStdDev, int batchSize, int tokenCount, int channelCount, float* δweight, float* δbias, float* δinput);
        unsafe void LayerNormForward(float* input, float* weight, float* bias, int batchSize, int tokenCount, int channelCount, float* mean, float* invStdDev, float* output);
        unsafe void MatMulBackward(float* δoutput, float* input, float* weight, int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount, float* δweight, float* δbias, float* δinput);
        unsafe void MatMulForward(float* input, float* weight, float* bias, int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount, float* output);
        unsafe void ResidualBackward(float* δoutput, int count, float* δleft, float* δright);
        unsafe void ResidualForward(float* left, float* right, int count, float* output);
        unsafe void SoftmaxForward(float* logits, int batchSize, int tokenCount, int vocabularySize, float* probabilities);
    }
    public class Llm : nietras.LargeLanguageModel.ILlm
    {
        public Llm() { }
        public virtual unsafe void AdamW(float* gradients, float* ms, float* vs, float* parameters, System.IntPtr parameterCount, float learningRate, float beta1, float beta2, float eps, float weightDecay, int t) { }
        public virtual unsafe void AttentionBackward(float* δoutput, float* postAttention, float* input, int batchSize, int tokenCount, int channelCount, int headCount, float* δpreAttention, float* δpostAttention, float* δinput) { }
        public virtual unsafe void AttentionForward(float* input, int batchSize, int tokenCount, int channelCount, int headCount, float* preAttention, float* postAttention, float* output) { }
        public virtual unsafe void CrossEntropyForward(float* probabilities, int* targetTokenIndices, int batchSize, int tokenCount, int vocabularySize, float* losses) { }
        public virtual unsafe void CrossEntropySoftmaxBackward(float* δlosses, float* probabilities, int* targetTokenIndices, int batchSize, int tokenCount, int vocabularySize, float* δlogits) { }
        public virtual unsafe void EmbedBackward(float* δoutput, int* tokenIndices, int batchSize, int tokenCount, int channelCount, float* δtokenEmbeddings, float* δpositionEmbeddings) { }
        public virtual unsafe void EmbedForward(int* tokenIndices, float* tokenEmbeddings, float* positionEmbeddings, int batchSize, int tokenCount, int channelCount, float* output) { }
        public virtual unsafe void GeLUBackward(float* δoutput, float* input, int count, float* δinput) { }
        public virtual unsafe void GeLUForward(float* input, int count, float* output) { }
        public virtual unsafe void LayerNormBackward(float* δoutput, float* input, float* weight, float* mean, float* invStdDev, int batchSize, int tokenCount, int channelCount, float* δweight, float* δbias, float* δinput) { }
        public virtual unsafe void LayerNormForward(float* input, float* weight, float* bias, int batchSize, int tokenCount, int channelCount, float* mean, float* invStdDev, float* output) { }
        public virtual unsafe void MatMulBackward(float* δoutput, float* input, float* weight, int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount, float* δweight, float* δbias, float* δinput) { }
        public virtual unsafe void MatMulForward(float* input, float* weight, float* bias, int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount, float* output) { }
        public virtual unsafe void ResidualBackward(float* δoutput, int count, float* δleft, float* δright) { }
        public virtual unsafe void ResidualForward(float* left, float* right, int count, float* output) { }
        public virtual unsafe void SoftmaxForward(float* logits, int batchSize, int tokenCount, int vocabularySize, float* probabilities) { }
    }
    public static class LlmFactory
    {
        public static System.Collections.Generic.IReadOnlyDictionary<string, System.Func<nietras.LargeLanguageModel.ILlm>> NameToCreate { get; }
        public static nietras.LargeLanguageModel.ILlm CreateDefault() { }
    }
    public class Llm_nietras : nietras.LargeLanguageModel.Llm
    {
        public Llm_nietras() { }
        public override unsafe void AdamW(float* gradients, float* ms, float* vs, float* parameters, System.IntPtr parameterCount, float learningRate, float beta1, float beta2, float eps, float weightDecay, int t) { }
        public override unsafe void AttentionBackward(float* δoutput, float* postAttention, float* input, int batchSize, int tokenCount, int channelCount, int headCount, float* δpreAttention, float* δpostAttention, float* δinput) { }
        public override unsafe void GeLUBackward(float* δoutput, float* input, int count, float* δinput) { }
        public override unsafe void GeLUForward(float* input, int count, float* output) { }
        public override unsafe void LayerNormBackward(float* δoutput, float* input, float* weight, float* mean, float* invStdDev, int batchSize, int tokenCount, int channelCount, float* δweight, float* δbias, float* δinput) { }
        public override unsafe void LayerNormForward(float* input, float* weight, float* bias, int batchSize, int tokenCount, int channelCount, float* mean, float* invStdDev, float* output) { }
        public override unsafe void MatMulBackward(float* δoutput, float* input, float* weight, int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount, float* δweight, float* δbias, float* δinput) { }
        public override unsafe void MatMulForward(float* input, float* weight, float* bias, int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount, float* output) { }
    }
    public static class Runner
    {
        public static void Run(string[] args, string dataDirectory, System.Action<string> log) { }
    }
}
```
