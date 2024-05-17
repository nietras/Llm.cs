using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace nietras.LargeLanguageModel;

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
