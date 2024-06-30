using System;
using System.Collections.Generic;

namespace nietras.LargeLanguageModel;

public static class LlmFactory
{
    public static string DefaultName { get; } = nameof(Llm_nietras);

    public static IReadOnlyDictionary<string, Func<ILlm>> NameToCreate { get; } =
        new Dictionary<string, Func<ILlm>>()
        {
            { nameof(Llm), () => new Llm() },
            { nameof(Llm_nietras), () => new Llm_nietras() },
        };
}
