using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace nietras.LargeLanguageModel;

internal static class Extensions
{
    public static unsafe void ReadExactlyUnmanaged<T>(this FileStream file, Span<T> values)
        where T : unmanaged
    {
        fixed (T* ptr = values)
        {
            ReadExactlyUnmanaged(file, ptr, values.Length);
        }
    }

    public static unsafe void ReadExactlyUnmanaged<T>(this FileStream file, T* values, long count)
        where T : unmanaged
    {
        Span<T> buffer = stackalloc T[(256 * 1024) / Unsafe.SizeOf<T>()];
        var totalReadCount = 0;
        while (totalReadCount < count)
        {
            var countToRead = (int)Math.Min(buffer.Length, count - totalReadCount);
            var bufferToRead = buffer.Slice(0, countToRead);
            var span = MemoryMarshal.Cast<T, byte>(bufferToRead);
            file.ReadExactly(span);
            bufferToRead.CopyTo(new Span<T>(values + totalReadCount, countToRead));
            totalReadCount += countToRead;
        }
    }

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
