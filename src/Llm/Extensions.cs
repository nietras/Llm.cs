using System.Collections.Generic;

namespace nietras.LargeLanguageModel;

internal static class Extensions
{
    public static IEnumerable<(int i0, int i1)> Enumerate(int count0, int count1)
    {
        for (var i0 = 0; i0 < count0; i0++)
        {
            for (var i1 = 0; i1 < count1; i1++)
            {
                yield return (i0, i1);
            }
        }
    }

    public static IEnumerable<(int i0, int i1, int i2)> Enumerate(int count0, int count1, int count2)
    {
        for (var i0 = 0; i0 < count0; i0++)
        {
            for (var i1 = 0; i1 < count1; i1++)
            {
                for (var i2 = 0; i2 < count2; i2++)
                {
                    yield return (i0, i1, i2);
                }
            }
        }
    }
}
