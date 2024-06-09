using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace nietras.LargeLanguageModel.Test;

[TestClass]
public class RunnerTest
{
    public static IEnumerable<object[]> LlmNames { get; } =
        [[nameof(Llm_nietras)]];
    //LlmFactory.NameToCreate.Keys.Select(n => new object[] { n });

    public TestContext? TestContext { get; set; }

    [TestMethod]
    [DynamicData(nameof(LlmNames))]
    public void RunnerTest_Run(string llmName)
    {
        //if (Environment.ProcessorCount < 8) { Assert.Inconclusive("Skipping Run as otherwise too slow"); }
        Runner.Run(args: [llmName], dataDirectory: "../../../",
            t => { TestContext?.WriteLine(t); Trace.WriteLine(t); });
    }
}
