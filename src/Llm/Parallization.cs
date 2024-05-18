using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace nietras.LargeLanguageModel;

static class LlmParallel
{
    internal static void ForRanges(int count0, int count1, Action<int, int> body)
    {
        //Parallel.ForEach(Extensions.Enumerate(count0, count1), t => body(t.i0, t.i1));
        Parallel.For(0, count0 * count1, v =>
        {
            var i0 = v / count1;
            var i1 = v % count1;
            body(i0, i1);
        });
    }

    internal static void ForRanges(int count0, int count1, int count2, Action<int, int, int> body)
    {
        //Parallel.ForEach(Extensions.Enumerate(count0, count1, count2), t => body(t.i0, t.i1, t.i2));
        Parallel.For(0, count0 * count1 * count2, v =>
        {
            var i0 = v / (count1 * count2);
            var i1 = (v / count2) % count1;
            var i2 = v % count2;
            body(i0, i1, i2);
        });
    }

    internal static void For(int fromInclusive, int toExclusive, Action<int> body)
    {
        Parallel.For(fromInclusive, toExclusive, body);
    }
}

static class NotParallel
{
    internal static void ForRanges(int count0, int count1, Action<int, int> body)
    {
        for (var i0 = 0; i0 < count0; i0++)
        {
            for (var i1 = 0; i1 < count1; i1++)
            {
                body(i0, i1);
            }
        }
    }

    internal static void ForRanges(int count0, int count1, int count2, Action<int, int, int> body)
    {
        for (var i0 = 0; i0 < count0; i0++)
        {
            for (var i1 = 0; i1 < count1; i1++)
            {
                for (var i2 = 0; i2 < count2; i2++)
                {
                    body(i0, i1, i2);
                }
            }
        }
    }

    internal static void For(int fromInclusive, int toExclusive, Action<int> body)
    {
        for (var i = fromInclusive; i < toExclusive; i++) { body(i); }
    }
}

public class FixedThreadCountTaskScheduler : TaskScheduler
{
    readonly int _threadCount;
    readonly ConcurrentQueue<Task> _taskQueue = new();
    readonly List<Thread> _threads = new();
    int _runningTasks = 0;
    bool _stopRequested = false;
    readonly AutoResetEvent _workAvailableEvent = new(false);

    public FixedThreadCountTaskScheduler(int threadCount)
    {
        if (threadCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(threadCount), "Thread count must be greater than zero.");
        }

        _threadCount = threadCount;

        for (var i = 0; i < _threadCount; i++)
        {
            var thread = new Thread(ExecuteTasks);
            thread.Start();
            _threads.Add(thread);
        }
    }

    protected override IEnumerable<Task> GetScheduledTasks()
    {
        return _taskQueue;
    }

    protected override void QueueTask(Task task)
    {
        _taskQueue.Enqueue(task);
        if (_runningTasks < _threadCount)
        {
            Interlocked.Increment(ref _runningTasks);
            _workAvailableEvent.Set();
        }
    }

    protected override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued)
    {
        if (Thread.CurrentThread.IsBackground || !taskWasPreviouslyQueued)
        {
            return false;
        }

        return TryExecuteTask(task);
    }

    protected override bool TryDequeue(Task task)
    {
        throw new NotImplementedException();
        //return _taskQueue.TryDequeue(out task);
    }

    private void ExecuteTasks()
    {
        while (!_stopRequested)
        {
            if (_taskQueue.TryDequeue(out var task))
            {
                TryExecuteTask(task);
            }
            else
            {
                Interlocked.Decrement(ref _runningTasks);
                if (_runningTasks == 0 && _taskQueue.IsEmpty)
                {
                    _workAvailableEvent.WaitOne();
                    Interlocked.Increment(ref _runningTasks);
                }
            }
        }
    }

    public void Stop()
    {
        _stopRequested = true;
        _workAvailableEvent.Set();
        foreach (var thread in _threads)
        {
            thread.Join();
        }
    }
}
