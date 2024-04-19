﻿# Llm.cs - C# port of @karpathy [llm.c](https://github.com/karpathy/llm.c)
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
        public static unsafe void AttentionBackward(float* dinp, float* dpreatt, float* datt, float* dout, float* inp, float* att, int B, int T, int C, int NH) { }
        public static unsafe void AttentionForward(float* output, float* preatt, float* att, float* inp, int B, int T, int C, int NH) { }
        public static unsafe void CrossEntropyForward(float* losses, float* probs, int* targets, int B, int T, int V) { }
        public static unsafe void CrossEntropySoftmaxBackward(float* dlogits, float* dlosses, float* probs, int* targets, int B, int T, int V) { }
        public static unsafe void EncoderBackward(float* dwte, float* dwpe, float* dout, int* inp, int B, int T, int C) { }
        public static unsafe void EncoderForward(float* output, int* inp, float* wte, float* wpe, int B, int T, int C) { }
        public static unsafe void GeLUBackward(float* dinp, float* inp, float* dout, int N) { }
        public static unsafe void GeLUForward(float* output, float* inp, int N) { }
        public static unsafe void LayerNormBackward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, float* mean, float* rstd, int B, int T, int C) { }
        public static unsafe void LayerNormForward(float* output, float* mean, float* rstd, float* inp, float* weight, float* bias, int B, int T, int C) { }
        public static unsafe void MatMulBackward(float* dinp, float* dweight, float* dbias, float* dout, float* inp, float* weight, int B, int T, int C, int OC) { }
        public static unsafe void MatMulForward(float* output, float* inp, float* weight, float* bias, int B, int T, int C, int OC) { }
        public static unsafe void ResidualBackward(float* dinp1, float* dinp2, float* dout, int N) { }
        public static unsafe void ResidualForward(float* output, float* inp1, float* inp2, int N) { }
        public static unsafe void SoftmaxForward(float* probs, float* logits, int B, int T, int V) { }
    }
}
```
