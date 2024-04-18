using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Reflection;
using nietras.LargeLanguageModel;

Action<string> log = t => { Console.WriteLine(t); Trace.WriteLine(t); };

var location = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
var dataDirectory = Path.Combine(location!, "../../../");

DownloadBinaryFilesIfNotExists(Gpt2.FileNames, Gpt2.RemoteUrl, dataDirectory, log);

Gpt2.Test(dataDirectory);
//Gpt2.Train(dataDirectory);

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
