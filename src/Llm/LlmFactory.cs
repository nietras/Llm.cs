using System;
using System.Collections.Generic;

namespace nietras.LargeLanguageModel;

public static class LlmFactory
{
    public static IReadOnlyDictionary<string, Func<ILlm>> NameToLlmCreate { get; } =
        new Dictionary<string, Func<ILlm>>()
        {
            { nameof(Llm), () => new Llm() },
            { nameof(Llm_nietras), () => new Llm_nietras() },
        };

    public static ILlm CreateDefault() => new Llm_nietras();
}
