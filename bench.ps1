#!/usr/bin/env pwsh
param (
    [string]$filter = "*"
)
dotnet run -c Release -f net8.0 --project src\Llm.Benchmarks\Llm.Benchmarks.csproj -- -m --warmupCount 1 --minIterationCount 1 --maxIterationCount 3 --runtimes net80 --iterationTime 300 --hide Method --filter $filter
