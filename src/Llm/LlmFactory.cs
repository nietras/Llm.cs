using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection;

namespace nietras.LargeLanguageModel;

public static class LlmFactory
{
    public static ILlm CreateDefault() => new Llm_nietras();

    [RequiresUnreferencedCode("Requires all ILlm instance to be available")]
    // If we are listing the types manually there isn't much point to using
    // reflection, but want to avoid having to change code for new
    // contributions... fix in future e.g. use a simple dumb test for keeping
    // the list. Best would be to have source generator of course...
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicMethods | DynamicallyAccessedMemberTypes.PublicConstructors, typeof(Llm))]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicMethods | DynamicallyAccessedMemberTypes.PublicConstructors, typeof(Llm_nietras))]
    public static IReadOnlyDictionary<string, Func<ILlm>> FindNameToLLmCreator()
    {
        var nameToLlm = Assembly.GetExecutingAssembly().GetTypes()
            .Where(t => IsValidLlm(t))
            .ToDictionary(t => t.Name, t => CreateFunc(t.GetConstructor(Type.EmptyTypes)!));
        return nameToLlm;

        static bool IsValidLlm(Type t) =>
            typeof(ILlm).IsAssignableFrom(t) &&
            !t.IsInterface && !t.IsAbstract &&
            t.GetConstructor(Type.EmptyTypes) != null;
    }

    static Func<ILlm> CreateFunc(ConstructorInfo constructorInfo) =>
        () => (ILlm)constructorInfo.Invoke([]);
}
