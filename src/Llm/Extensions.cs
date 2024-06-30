using System;
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

    public static unsafe void ReadExactlyUnmanaged<T>(this FileStream file, T* values, nint count)
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
}
