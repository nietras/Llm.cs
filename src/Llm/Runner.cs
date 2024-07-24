using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;

namespace nietras.LargeLanguageModel;

public static class Runner
{
    public static void Run(string[] args, string dataDirectory, Action<string> log)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(dataDirectory);
        ArgumentNullException.ThrowIfNull(log);

        log($"{nameof(Environment.ProcessorCount)}: {Environment.ProcessorCount}");

        // Download the model and tokenizer files if they don't exist
        DownloadBinaryFilesIfNotExists(Gpt2.FileNames, Gpt2.RemoteUrl, dataDirectory, log);

        var name = args?.Length > 0 ? args[0] : LlmFactory.DefaultName;
        var llm = LlmFactory.NameToCreate[name]();

        // Log to file too for reference
        var logFilePath = Path.Combine(dataDirectory, $"{name}.log");
        using var logWriter = new StreamWriter(logFilePath);
        Action<string> newLog = t => { log(t); logWriter.WriteLine(t); };

        //Gpt2.Infer(dataDirectory, llm, newLog);
        const int steps = 10;
        Gpt2.VerifyTrain(dataDirectory, llm, steps, newLog);
        //Gpt2.Train(dataDirectory, llm);
    }

    internal static void DownloadBinaryFilesIfNotExists(
        IReadOnlyList<string> fileNames, Func<string, string> toUrl,
        string dataDirectory, Action<string>? log)
    {
        foreach (var fileName in fileNames)
        {
            var filePath = Path.Combine(dataDirectory, fileName);
            filePath = Path.GetFullPath(filePath);
            if (!File.Exists(filePath))
            {
                var url = toUrl(fileName);
                log?.Invoke($"Downloading '{url}' to '{filePath}'");
                using var client = new HttpClient();
                // Download the file
                var source = client.GetStreamAsync(url).Result;
                using var destination = new FileStream(filePath, FileMode.Create);
                source.CopyTo(destination);
            }
        }
    }
}
