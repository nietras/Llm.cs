# llm.cs - C# port of @karpathy [llm.c](https://github.com/karpathy/llm.c)
![.NET](https://img.shields.io/badge/net8.0-5C2D91?logo=.NET&labelColor=gray)
![C#](https://img.shields.io/badge/12.0-239120?logo=csharp&logoColor=white&labelColor=gray)
![Lines of code](https://tokei.rs/b1/github/nietras/llm.cs?category=code)
[![Build Status](https://github.com/nietras/llm.cs/actions/workflows/dotnet.yml/badge.svg?branch=main)](https://github.com/nietras/llm.cs/actions/workflows/dotnet.yml)
[![Super-Linter](https://github.com/nietras/llm.cs/actions/workflows/super-linter.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![codecov](https://codecov.io/gh/nietras/llm.cs/branch/main/graph/badge.svg?token=WN56CR3X0D)](https://codecov.io/gh/nietras/llm.cs)
[![CodeQL](https://github.com/nietras/llm.cs/workflows/CodeQL/badge.svg)](https://github.com/nietras/llm.cs/actions?query=workflow%3ACodeQL)
[![Nuget](https://img.shields.io/nuget/v/Llm?color=purple)](https://www.nuget.org/packages/Llm/)
[![Release](https://img.shields.io/github/v/release/nietras/llm.cs)](https://github.com/nietras/llm.cs/releases/)
[![downloads](https://img.shields.io/nuget/dt/Llm)](https://www.nuget.org/packages/Llm)
![Size](https://img.shields.io/github/repo-size/nietras/Llm.cs.svg)
[![License](https://img.shields.io/github/license/nietras/Llm.cs)](https://github.com/nietras/llm.cs/blob/main/LICENSE)
[![Blog](https://img.shields.io/badge/blog-nietras.com-4993DD)](https://nietras.com)

⭐ Please star this project if you like it. ⭐

TODO

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
    public static class Gpt2
    {
        public static void Train() { }
        public static unsafe void attention_backward(float* dinp, float* dpreatt, float* datt, float* dout, float* inp, float* att, int B, int T, int C, int NH) { }
        public static unsafe void attention_forward(float* output, float* preatt, float* att, float* inp, int B, int T, int C, int NH) { }
        public static unsafe T* calloc<T>(long size)
            where T :  unmanaged { }
        public static unsafe void crossentropy_forward(float* losses, float* probs, int* targets, int B, int T, int V) { }
        public static unsafe void crossentropy_softmax_backward(float* dlogits, float* dlosses, float* probs, int* targets, int B, int T, int V) { }
        public static unsafe void encoder_backward(float* dwte, float* dwpe, float* dout, int* inp, int B, int T, int C) { }
        public static unsafe void encoder_forward(float* output, int* inp, float* wte, float* wpe, int B, int T, int C) { }
        public static unsafe void free<T>(T* ptr)
            where T :  unmanaged { }
        public static unsafe void gelu_backward(float* dinp, float* inp, float* dout, int N) { }
        public static unsafe void gelu_forward(float* output, float* inp, int N) { }
        public static unsafe void gpt2_build_from_checkpoint(nietras.LargeLanguageModel.Gpt2.GPT2* model, string checkpoint_path) { }
        public static unsafe void layernorm_backward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, float* mean, float* rstd, int B, int T, int C) { }
        public static unsafe void layernorm_forward(float* output, float* mean, float* rstd, float* inp, float* weight, float* bias, int B, int T, int C) { }
        public static unsafe T* malloc<T>(long size)
            where T :  unmanaged { }
        public static unsafe float* malloc_and_point_activations(nietras.LargeLanguageModel.Gpt2.ActivationTensors* acts, long* act_sizes) { }
        public static unsafe float* malloc_and_point_parameters(nietras.LargeLanguageModel.Gpt2.ParameterTensors* parameters, long* param_sizes) { }
        public static unsafe void matmul_backward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, int B, int T, int C, int OC) { }
        public static unsafe void matmul_forward(float* output, float* inp, float* weight, float* bias, int B, int T, int C, int OC) { }
        public static unsafe void memcpy<T>(T* dest, T* src, long size)
            where T :  unmanaged { }
        public static unsafe void memset<T>(T* ptr, long size)
            where T :  unmanaged { }
        public static unsafe void residual_backward(float* dinp1, float* dinp2, float* dout, int N) { }
        public static unsafe void residual_forward(float* output, float* inp1, float* inp2, int N) { }
        public static unsafe void softmax_forward(float* probs, float* logits, int B, int T, int V) { }
        public struct ActivationTensors
        {
            public unsafe float* att;
            public unsafe float* attproj;
            public unsafe float* atty;
            public unsafe float* encoded;
            public unsafe float* fch;
            public unsafe float* fch_gelu;
            public unsafe float* fcproj;
            public unsafe float* ln1;
            public unsafe float* ln1_mean;
            public unsafe float* ln1_rstd;
            public unsafe float* ln2;
            public unsafe float* ln2_mean;
            public unsafe float* ln2_rstd;
            public unsafe float* lnf;
            public unsafe float* lnf_mean;
            public unsafe float* lnf_rstd;
            public unsafe float* logits;
            public unsafe float* losses;
            public unsafe float* preatt;
            public unsafe float* probs;
            public unsafe float* qkv;
            public unsafe float* residual2;
            public unsafe float* residual3;
        }
        public class DataLoader : System.IDisposable
        {
            public int B;
            public int T;
            public unsafe int* batch;
            public long current_position;
            public long file_size;
            public unsafe int* inputs;
            public long num_batches;
            public unsafe int* targets;
            public System.IO.FileStream tokens_file;
            public DataLoader(string filename, int B, int T) { }
            public void Dispose() { }
            protected virtual void Dispose(bool disposing) { }
            public void dataloader_free() { }
            public void dataloader_next_batch() { }
            public void dataloader_reset() { }
        }
        public struct GPT2
        {
            [System.Runtime.CompilerServices.FixedBuffer(typeof(long), 23)]
            public nietras.LargeLanguageModel.Gpt2.GPT2.<act_sizes>e__FixedBuffer act_sizes;
            public nietras.LargeLanguageModel.Gpt2.ActivationTensors acts;
            public unsafe float* acts_memory;
            public int batch_size;
            public nietras.LargeLanguageModel.Gpt2.GPT2Config config;
            public nietras.LargeLanguageModel.Gpt2.ParameterTensors grads;
            public nietras.LargeLanguageModel.Gpt2.ActivationTensors grads_acts;
            public unsafe float* grads_acts_memory;
            public unsafe float* grads_memory;
            public unsafe int* inputs;
            public unsafe float* m_memory;
            public float mean_loss;
            public long num_activations;
            public long num_parameters;
            [System.Runtime.CompilerServices.FixedBuffer(typeof(long), 16)]
            public nietras.LargeLanguageModel.Gpt2.GPT2.<param_sizes>e__FixedBuffer param_sizes;
            public nietras.LargeLanguageModel.Gpt2.ParameterTensors parameters;
            public unsafe float* params_memory;
            public int seq_len;
            public unsafe int* targets;
            public unsafe float* v_memory;
        }
        public struct GPT2Config
        {
            public int channels;
            public int max_seq_len;
            public int num_heads;
            public int num_layers;
            public int vocab_size;
        }
        public struct ParameterTensors
        {
            public unsafe float* attprojb;
            public unsafe float* attprojw;
            public unsafe float* fcb;
            public unsafe float* fcprojb;
            public unsafe float* fcprojw;
            public unsafe float* fcw;
            public unsafe float* ln1b;
            public unsafe float* ln1w;
            public unsafe float* ln2b;
            public unsafe float* ln2w;
            public unsafe float* lnfb;
            public unsafe float* lnfw;
            public unsafe float* qkvb;
            public unsafe float* qkvw;
            public unsafe float* wpe;
            public unsafe float* wte;
        }
    }
}
```
