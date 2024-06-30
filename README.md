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
  ```powershell
  [GPT-2]
  max_seq_len: 1024
  vocab_size: 50257
  num_layers: 12
  num_heads: 12
  channels: 768
  num_parameters: 124439808
  [State]
  batch_size: 4
  seq_len: 64
  activationCount: 73323776
  Logits           TENSOR OK
  dwte             TENSOR OK
  dwpe             TENSOR OK
  dln1w            TENSOR OK
  dln1b            TENSOR OK
  dqkvw            TENSOR OK
  dqkvb            TENSOR OK
  dattprojw        TENSOR OK
  dattprojb        TENSOR OK
  dln2w            TENSOR OK
  dln2b            TENSOR OK
  dfcw             TENSOR OK
  dfcb             TENSOR OK
  dfcprojw         TENSOR OK
  dfcprojb         TENSOR OK
  dlnfw            TENSOR OK
  dlnfb            TENSOR OK
  step 0: loss 5.269890 expected loss 5.270007 OK   (took 4219 ms)
  step 1: loss 4.059388 expected loss 4.059707 OK   (took 4099 ms)
  step 2: loss 3.374212 expected loss 3.375123 OK   (took 4050 ms)
  step 3: loss 2.800128 expected loss 2.800783 OK   (took 4073 ms)
  step 4: loss 2.315312 expected loss 2.315382 OK   (took 4089 ms)
  step 5: loss 1.849347 expected loss 1.849029 OK   (took 4052 ms)
  step 6: loss 1.395217 expected loss 1.394656 OK   (took 4071 ms)
  step 7: loss 0.998616 expected loss 0.999147 OK   (took 4057 ms)
  step 8: loss 0.625540 expected loss 0.624080 OK   (took 4073 ms)
  step 9: loss 0.378012 expected loss 0.376511 OK   (took 4059 ms)
  overall okay: True
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
