using System;
using System.Diagnostics.Contracts;

namespace nietras.LargeLanguageModel;

public readonly record struct Llm
{
    const char _min = (char)32;
    const char _max = (char)126;

    readonly char _separator;

    public Llm() : this(LlmDefaults.Llmarator) { }

    public Llm(char separator)
    {
        Validate(separator);
        _separator = separator;
    }

    public char Llmarator
    {
        get => _separator;
        init { Validate(value); _separator = value; }
    }

    public static Llm Default { get; } = new(LlmDefaults.Llmarator);
    public static Llm? Auto => null;

    internal static Llm Min { get; } = new(_min);
    internal static Llm Max { get; } = new(_max);

    public static Llm New(char separator) => new(separator);

    public static LlmReaderOptions Reader() => new(null);
    public static LlmReaderOptions Reader(Func<LlmReaderOptions, LlmReaderOptions> configure)
    {
        Contract.Assume(configure != null);
        return configure(Reader());
    }

    public static LlmWriterOptions Writer() => new(Default);
    public static LlmWriterOptions Writer(Func<LlmWriterOptions, LlmWriterOptions> configure)
    {
        Contract.Assume(configure != null);
        return configure(Writer());
    }

    internal static void Validate(char separator)
    {
        if (separator != '\t' && (separator < _min || separator > _max))
        {
            LlmThrow.ArgumentOutOfRangeException_Llmarator(separator);
        }
        if (separator == LlmDefaults.Comment ||
            separator == LlmDefaults.Quote)
        {
            LlmThrow.ArgumentException_Llmarator(separator);
        }
    }

    internal string[] Split(string line) => line.Split(Llmarator, StringSplitOptions.None);
}
