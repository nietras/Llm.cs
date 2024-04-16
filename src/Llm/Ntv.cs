using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace nietras.LargeLanguageModel;

[DebuggerDisplay("Count = {Count} ByteCount = {ByteCount} Ptr = {Ptr:X}")]
internal unsafe class Ntv<T> : SafeHandleZeroOrMinusOneIsInvalid
    where T : unmanaged
{
    const int Alignment = 64;

    public Ntv(long count) : base(true)
    {
        Ptr = (T*)NativeMemory.AlignedAlloc((nuint)(count * sizeof(T)), Alignment);
        Count = count;
        SetHandle(new IntPtr(Ptr));
    }

    public T* Ptr { get; private set; }
    public long Count { get; }
    public nuint ByteCount => (nuint)(Count * sizeof(T));

    public Span<T> DebugSpan => new(Ptr, (int)(Math.Min(int.MaxValue, Count)));

    public void Clear() => NativeMemory.Clear(Ptr, ByteCount);

    protected override bool ReleaseHandle()
    {
        NativeMemory.AlignedFree(Ptr);
        Ptr = null;
        return true;
    }
}
