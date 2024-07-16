using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace nietras.LargeLanguageModel;

public interface ITensor
{
    public ReadOnlySpan<nint> Lengths { get; }
    public ReadOnlySpan<nint> Strides { get; }
    public nint Count { get; }
    public nint ByteCount { get; }
    public string Name { get; }
}

public unsafe interface ITensor<T> : ITensor where T : unmanaged
{
    public T* Ptr { get; }
    public T* StrideToPtrAt(int index) => Ptr + Strides[0] * index;
}

public unsafe delegate T* GetPtr<T>() where T : unmanaged;
[DebuggerDisplay("{DebuggerDisplay,nq}")]
public sealed unsafe class Tensor<T>(GetPtr<T> ptr, nint offset, nint[] lengths, nint[] strides, string name)
    : ITensor<T>
    where T : unmanaged
{
    public T* Ptr => ptr() + Offset;
    public T* StrideToPtrAt(int index) => Ptr + Strides[0] * index;
    public nint Offset { get; } = offset;
    public ReadOnlySpan<nint> Lengths => lengths;
    public ReadOnlySpan<nint> Strides => strides;
    public nint Count { get; } = lengths.Product();
    public nint ByteCount { get; } = lengths.Product() * sizeof(T);
    public string Name { get; } = name;

    public Span<T> DebugSpan => new(Ptr, (int)(Math.Min(int.MaxValue, Count)));

#pragma warning disable CA1062 // Validate arguments of public methods
    public static implicit operator T*(Tensor<T> t) => t.Ptr;
#pragma warning restore CA1062 // Validate arguments of public methods

    string DebuggerDisplay => $"{Name} {Lengths.ToShapeText()}={Count:D}";
}

public unsafe class Tensors<T>(object s)
    : IReadOnlyList<Tensor<T>>, IDisposable
    where T : unmanaged
{
    readonly State _state = (State)s;

    public static TTensors Create<TTensors>(Func<object, TTensors> create)
    {
        ArgumentNullException.ThrowIfNull(create);
        var s = new State();
        var tensors = create(s);
        s.Ntv = new Ntv<T>(s.TotalCount);
        return tensors;
    }

    public T* MemoryPtr => _state.Ntv!.Ptr;

    public nint TotalCount => _state.TotalCount;

    public int Count => _state.Tensors.Count;

    public Tensor<T> this[int index] => _state.Tensors[index];

    public IEnumerator<Tensor<T>> GetEnumerator() => _state.Tensors.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => _state.Tensors.GetEnumerator();

    internal unsafe static Tensor<T> New(nint[] lengths, object s, [CallerMemberName] string name = "")
    {
        var state = (State)s;
        var strides = lengths.CalculateStrides();
        Tensor<T> tensor = new(() => state.Ntv!.Ptr, state.TotalCount, lengths, strides, name);
        state.Tensors.Add(tensor);
        state.TotalCount += tensor.Count;
        return tensor;
    }

    internal sealed class State
    {
        internal Ntv<T>? Ntv { get; set; } = null;
        internal List<Tensor<T>> Tensors { get; set; } = [];
        internal nint TotalCount { get; set; } = 0;
    }

    void DisposeManagedResources()
    {
        _state.Ntv?.Dispose();
    }

    #region Dispose
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    // Dispose(bool disposing) executes in two distinct scenarios.
    // If disposing equals true, the method has been called directly
    // or indirectly by a user's code. Managed and unmanaged resources
    // can be disposed.
    // If disposing equals false, the method has been called by the 
    // runtime from inside the finalizer and you should not reference 
    // other objects. Only unmanaged resources can be disposed.
    protected virtual void Dispose(bool disposing)
    {
        // Dispose only if we have not already disposed.
        if (!m_disposed)
        {
            // If disposing equals true, dispose all managed and unmanaged resources.
            // I.e. dispose managed resources only if true, unmanaged always.
            if (disposing)
            {
                DisposeManagedResources();
            }

            // Call the appropriate methods to clean up unmanaged resources here.
            // If disposing is false, only the following code is executed.
        }
        m_disposed = true;
    }
    volatile bool m_disposed = false;
    #endregion
}


[DebuggerDisplay("Count = {Count} ByteCount = {ByteCount} Ptr = {IntPtr}")]
unsafe class Ntv<T> : SafeHandleZeroOrMinusOneIsInvalid
    where T : unmanaged
{
    const int Alignment = 64;

    public Ntv(nint count) : base(true)
    {
        Count = count;
        ByteCount = Count * sizeof(T);
        Ptr = (T*)NativeMemory.AlignedAlloc((nuint)ByteCount, Alignment);
        SetHandle(new IntPtr(Ptr));
    }

    public nint Count { get; }
    public nint ByteCount { get; }
    public T* Ptr { get; private set; }
    public IntPtr IntPtr => new(Ptr);

    public Span<T> DebugSpan => new(Ptr, (int)(Math.Min(int.MaxValue, Count)));

    public void Clear() => NativeMemory.Clear(Ptr, (nuint)ByteCount);

    protected override bool ReleaseHandle()
    {
        NativeMemory.AlignedFree(Ptr);
        Ptr = null;
        return true;
    }
}
