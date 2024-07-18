using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace nietras.LargeLanguageModel;

static partial class Gpt2
{
    public static unsafe void Infer(string dataDirectory, ILlm llmToUse,
        Action<string>? log)
    {
        // build the GPT-2 model from a checkpoint
        using var model = ModelFromCheckpoint(dataDirectory + ModelBinaryFileName);
        // Arbitrarily limit max tokens here for now
        var maxTokenCount = Math.Min(256, model.Config.MaxTokenCount);

        var tokenizer = Bpe.ReadGpt2FromTiktokenFile(dataDirectory + TokenizerTiktokenFileName);

        var stopwatch = new Stopwatch();
        var llm = CreateTimeLlm(llmToUse);
        var promptTokenIndices = new List<int>();

        int* tokenIndices = stackalloc int[maxTokenCount];
        var tokenIndicesSpan = new Span<int>(tokenIndices, maxTokenCount);
        int tokenCount = 0;

        // some memory for generating samples from the model
        ulong randomNumberState = 1337;

        while (true)
        {
            LogNoNewLine("Prompt: ");
            var line = Console.ReadLine();
            if (line == null) { continue; }

            promptTokenIndices.Clear();
            tokenizer.Encode(line, promptTokenIndices);

            var tokensToCopy = Math.Min(maxTokenCount - 1, promptTokenIndices.Count);
            CollectionsMarshal.AsSpan(promptTokenIndices).Slice(0, tokensToCopy).CopyTo(tokenIndicesSpan);
            tokenIndicesSpan[tokensToCopy] = EndOfTextTokenIndex;
            tokenCount = tokensToCopy + 1;

            Log($"Prompt (encode-decode): {tokenizer.TryDecode(tokenIndicesSpan.Slice(0, tokenCount))}");
            while (tokenCount < maxTokenCount)
            {
                // note that inference is wasteful here because for each t,
                // we re-compute all activations between 0 and t leaving
                // this alone because you want separate code for inference
                // anyway the inference here is just for sanity checking
                // purposes
                Forward(model, tokenIndices, null, 1, tokenCount, llm, maxTokenCount);

                float* probabilities = model.Outputs!.Probabilities.Ptr + (tokenCount - 1) * model.Config.VocabularySize;
                float coin = RandomSingle(&randomNumberState);
                int nextToken = FindSampleIndex(probabilities, model.Config.VocabularySize, coin);
                tokenIndices[tokenCount] = nextToken;
                ++tokenCount;
                var output = tokenizer.TryDecode([nextToken]) ?? string.Empty;
                LogNoNewLine(output);
            }
            Log(string.Empty);
        }
    }
}
