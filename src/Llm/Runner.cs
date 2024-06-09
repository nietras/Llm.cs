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

        // download the model and tokenizer files if they don't exist
        DownloadBinaryFilesIfNotExists(Gpt2.FileNames, Gpt2.RemoteUrl, dataDirectory, log);

        ILlm llm = (args?.Length > 0 && LlmFactory.NameToCreate.TryGetValue(args[0], out var create))
            ? create() : LlmFactory.CreateDefault();
        Gpt2.Test(dataDirectory, llm);
        //Gpt2.Train(dataDirectory);
    }

    static void DownloadBinaryFilesIfNotExists(
        IReadOnlyList<string> fileNames, Func<string, string> toUrl,
        string dataDirectory, Action<string> log)
    {
        foreach (var fileName in fileNames)
        {
            var filePath = Path.Combine(dataDirectory, fileName);
            filePath = Path.GetFullPath(filePath);
            if (!File.Exists(filePath))
            {
                var url = toUrl(fileName);
                log($"Downloading '{url}' to '{filePath}'");
                using var client = new HttpClient();
                // Download the file
                var source = client.GetStreamAsync(url).Result;
                using var destination = new FileStream(filePath, FileMode.Create);
                source.CopyTo(destination);
            }
        }
    }
}
