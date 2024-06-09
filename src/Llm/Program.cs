using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using nietras.LargeLanguageModel;

Action<string> log = t => { Console.WriteLine(t); Trace.WriteLine(t); };

var location = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
var dataDirectory = Path.Combine(location!, "../../../");

Runner.Run(args, dataDirectory, log);
