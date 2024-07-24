using System.Collections.Generic;
using System.Diagnostics;
using System.IO;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

static partial class Gpt2
{
    internal const string ModelBinaryFileName = "gpt2_124M.bin";
    internal const string ModelDebugBinaryFileName = "gpt2_124M_debug_state.bin";

    internal const string TokenizerTiktokenFileName = "gpt2.tiktoken";

    internal const string DataTinyStoriesTrainBinaryFileName = "TinyStories_train.bin";
    internal const string DataTinyStoriesValidationBinaryFileName = "TinyStories_val.bin";

    internal const string TinyShakespeareTrainBinaryFileName = "tiny_shakespeare_train.bin";
    internal const string TinyShakespeareValidationBinaryFileName = "tiny_shakespeare_val.bin";

    internal static readonly IReadOnlyList<string> FileNames = [
        ModelBinaryFileName,
        ModelDebugBinaryFileName,
        //DataTinyStoriesTrainBinaryFileName,
        //DataTinyStoriesValidationBinaryFileName,
        TinyShakespeareTrainBinaryFileName,
        TinyShakespeareValidationBinaryFileName];

    internal static string RemoteUrl(string fileName) =>
        @$"https://huggingface.co/datasets/nietras/llm.bin/resolve/main/{fileName}?download=true";

    // ----------------------------------------------------------------------------
    // main training loop
    public static unsafe void Train(string dataDirectory, ILlm llmToUse)
    {
        // build the GPT-2 model from a checkpoint
        using var model = ModelFromCheckpoint(dataDirectory + ModelBinaryFileName);

        var tokenizer = Bpe.ReadGpt2FromTiktokenFile(dataDirectory + TokenizerTiktokenFileName);

        // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
        var tinyStoriesTrain = dataDirectory + DataTinyStoriesTrainBinaryFileName;
        var tinyStoriesValidation = dataDirectory + DataTinyStoriesValidationBinaryFileName;
        var tinyShakespeareTrain = dataDirectory + TinyShakespeareTrainBinaryFileName;
        var tinyShakespeareValidation = dataDirectory + TinyShakespeareValidationBinaryFileName;
        var trainTokens = File.Exists(tinyShakespeareTrain) ? tinyShakespeareTrain : tinyStoriesTrain;
        var valTokens = File.Exists(tinyShakespeareValidation) ? tinyShakespeareValidation : tinyStoriesValidation;
        int b = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        int t = 64; // sequence length 64 (i.e. each sequence is 64 tokens int). must be <= maxT, which is 1024 for GPT-2
        using DataLoader trainLoader = new(trainTokens, b, t);
        Log($"Train dataset BatchCount: {trainLoader.BatchCount}");

        using DataLoader validLoader = new(valTokens, b, t);
        Log($"Valid dataset BatchCount: {validLoader.BatchCount}");
        int validBatchCount = 10;

        // some memory for generating samples from the model
        ulong randomNumberState = 1337;
        // during inference step we'll generate sequences of this many tokens
        const int maxGeneratedTokenCount = 64;
        int* generatedTokens = stackalloc int[maxGeneratedTokenCount];

        // train
        var stopwatch = new Stopwatch();
        var llm = CreateTimeLlm(llmToUse);
        for (int step = 0; step <= 20; step++)
        {
            // do a training step
            // TODO: Abstract loader and add to step perhaps and part of timings)
            trainLoader.NextBatch();
            var (loss, timings) = TrainStep(model, trainLoader.InputTokenIndices,
                trainLoader.TargetTokenIndices, b, t, llm, step);
            Log($"{step:D2}: train loss {loss:F6} ({timings.ToReport()})");

            // once in a while estimate the validation loss
            if (step % 10 == 0)
            {
                float validLoss = 0.0f;
                validLoader.Reset();
                for (int i = 0; i < validBatchCount; i++)
                {
                    validLoader.NextBatch();
                    var validBatchLoss = Forward(model, validLoader.InputTokenIndices,
                        validLoader.TargetTokenIndices, b, t, llm);
                    validLoss += validBatchLoss;
                }
                validLoss /= validBatchCount;
                Log($"Valid loss: {validLoss}");
            }

            // once in a while do model inference to print generated text
            if (step % 10 == 0)
            {
                // the GPT-2 EOT token kicks off the generation
                generatedTokens[0] = EndOfTextTokenIndex;
                for (int ti = 1; ti < maxGeneratedTokenCount; ti++)
                {
                    // note that inference is wasteful here because for each t,
                    // we re-compute all activations between 0 and t leaving
                    // this alone because you want separate code for inference
                    // anyway the inference here is just for sanity checking
                    // purposes
                    Forward(model, generatedTokens, null, 1, ti, llm);
                    float* probabilities = model.Outputs!.Probabilities.Ptr + (ti - 1) * model.Config.VocabularySize;
                    float coin = RandomSingle(&randomNumberState);
                    int nextToken = FindSampleIndex(probabilities, model.Config.VocabularySize, coin);
                    generatedTokens[ti] = nextToken;
                }
                LogNoNewLine("generated: [");
                for (int ti = 0; ti < maxGeneratedTokenCount; ti++)
                {
                    LogNoNewLine($"{generatedTokens[ti]} ");
                }
                Log("]");
                Log($"generated: '{tokenizer.TryDecode(new(generatedTokens, maxGeneratedTokenCount))}'");
            }
        }
    }
}
