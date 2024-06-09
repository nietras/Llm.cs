using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace nietras.LargeLanguageModel.Test;

[TestClass]
public class RunnerTest
{
    public static IEnumerable<object[]> LlmNames { get; } =
        [[nameof(Llm_nietras)]];
    //LlmFactory.NameToCreate.Keys.Select(n => new object[] { n });

    // Only run in Release since may be too slow in Debug
#if RELEASE
    [ExcludeFromCodeCoverage]
    [TestMethod]
    [DynamicData(nameof(LlmNames))]
#endif
    public void RunnerTest_Run(string llmName)
    {
        Trace.WriteLine("TEST TEST");
        //if (Environment.ProcessorCount < 8) { Assert.Inconclusive("Skipping Run as otherwise too slow"); }
        Runner.Run(args: [llmName], dataDirectory: "../../../",
            t => Trace.WriteLine(t));
    }
}
