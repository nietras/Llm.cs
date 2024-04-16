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
    public static class Llm
    {
        public static unsafe void attention_backward(float* dinp, float* dpreatt, float* datt, float* dout, float* inp, float* att, int B, int T, int C, int NH) { }
        public static unsafe void attention_forward(float* output, float* preatt, float* att, float* inp, int B, int T, int C, int NH) { }
        public static unsafe void crossentropy_forward(float* losses, float* probs, int* targets, int B, int T, int V) { }
        public static unsafe void crossentropy_softmax_backward(float* dlogits, float* dlosses, float* probs, int* targets, int B, int T, int V) { }
        public static unsafe void encoder_backward(float* dwte, float* dwpe, float* dout, int* inp, int B, int T, int C) { }
        public static unsafe void encoder_forward(float* output, int* inp, float* wte, float* wpe, int B, int T, int C) { }
        public static unsafe void gelu_backward(float* dinp, float* inp, float* dout, int N) { }
        public static unsafe void gelu_forward(float* output, float* inp, int N) { }
        public static unsafe void layernorm_backward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, float* mean, float* rstd, int B, int T, int C) { }
        public static unsafe void layernorm_forward(float* output, float* mean, float* rstd, float* inp, float* weight, float* bias, int B, int T, int C) { }
        public static unsafe void matmul_backward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, int B, int T, int C, int OC) { }
        public static unsafe void matmul_forward(float* output, float* inp, float* weight, float* bias, int B, int T, int C, int OC) { }
        public static unsafe void residual_backward(float* dinp1, float* dinp2, float* dout, int N) { }
        public static unsafe void residual_forward(float* output, float* inp1, float* inp2, int N) { }
        public static unsafe void softmax_forward(float* probs, float* logits, int B, int T, int V) { }
    }
}
```
