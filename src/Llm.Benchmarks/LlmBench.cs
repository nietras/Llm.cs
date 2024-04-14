using BenchmarkDotNet.Attributes;

namespace nietras.LargeLanguageModel.Benchmarks;

public class LlmBench
{
    int i = 1;

    [GlobalSetup]
    public void GlobalSetup()
    {
        i = 3;
    }

    [Benchmark(Baseline = true)]
    public int Naive() => i;
}
