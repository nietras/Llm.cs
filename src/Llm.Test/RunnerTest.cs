using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace nietras.LargeLanguageModel.Test;

[TestClass]
public class RunnerTest
{
    public static IEnumerable<object[]> LlmNames { get; } =
        LlmFactory.NameToCreate.Keys.Select(n => new object[] { n });

    [TestMethod]
    [DynamicData(nameof(LlmNames))]
    public void RunnerTest_Run(string llmName)
    {
        Runner.Run(args: [llmName], dataDirectory: "../../../",
            t => Trace.WriteLine(t));
    }
}
