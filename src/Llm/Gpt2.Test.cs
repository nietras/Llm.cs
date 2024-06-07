using System;
using System.Diagnostics;
using System.IO;

namespace nietras.LargeLanguageModel;

#pragma warning disable IDE0007 // Use implicit type

internal static partial class Gpt2
{
    public static unsafe void Test(string dataDirectory)
    {
        // build the GPT-2 model from a checkpoint
        GPT2 model;
        BuildFromCheckpoint(&model, dataDirectory + ModelBinaryFileName);

        int C = model.config.channels;
        int V = model.config.vocab_size;
        int maxT = model.config.max_seq_len;
        int L = model.config.num_layers;

        using var state_file = File.OpenRead(dataDirectory + ModelDebugBinaryFileName);
        Span<int> state_header = stackalloc int[256];
        // read span from model_file
        state_file.ReadExactlyUnmanaged(state_header);
        //fread(model_header, sizeof(int), 256, model_file);
        if (state_header[0] != 20240327) { throw new InvalidDataException($"Bad magic model file"); }
        if (state_header[1] != 1) { throw new InvalidDataException($"Bad version in model file"); }

        int B = state_header[2]; // batch size, e.g. 4
        int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
        Log("[State]");
        Log($"batch_size: {B}");
        Log($"seq_len: {T}");

        ParameterTensors expected_grads;
        float* expected_grads_memory = AllocateAndPointParameters(&expected_grads, model.param_sizes);

        // inputs and expected outputs, only used for error checking
        int* x = malloc<int>(B * T);
        int* y = malloc<int>(B * T);
        float* expected_logits = malloc<float>(B * T * V);
        float* expected_loss = malloc<float>(1);

        // read reference information from Python
        state_file.ReadExactlyUnmanaged(x, B * T);
        state_file.ReadExactlyUnmanaged(y, B * T);
        state_file.ReadExactlyUnmanaged(expected_logits, B * T * V);
        state_file.ReadExactlyUnmanaged(expected_loss, 1);
        state_file.ReadExactlyUnmanaged(expected_grads_memory, model.num_parameters);
        state_file.Dispose();

        // expected losses are as follows, from Python
        float[] expected_losses = {
            5.270007133483887f,
            4.059706687927246f,
            3.3751230239868164f,
            2.8007826805114746f,
            2.315382242202759f,
            1.8490285873413086f,
            1.3946564197540283f,
            0.9991465210914612f,
            0.6240804195404053f,
            0.37651097774505615f
        };

        // overall OK signal for the test
        bool allOk = true;

        // let's do 10 training iterations, following the pytorch code
        const int steps = 10;
        float* losses = stackalloc float[steps];
        var stopwatch = new Stopwatch();
        var llm = new TimeLlm(new Llm());
        for (int step = 0; step < steps; step++)
        {
            stopwatch.Restart();

            Forward(&model, x, y, B, T, llm);
            double t1_ms = stopwatch.Elapsed.TotalMilliseconds;
            ZeroGrad(&model, llm);
            double t2_ms = stopwatch.Elapsed.TotalMilliseconds;
            Backward(&model, llm);
            double t3_ms = stopwatch.Elapsed.TotalMilliseconds;

            if (step == 0)
            {
                // error checking at step 0 for reference activations/gradients

                // at this point, target should be equal to expected_logits, let's compare
                allOk &= CheckTensor(expected_logits, model.acts.logits, B * T * V, "Logits");

                // finally check all the gradients
                var gradoks = new bool[16];
                ParameterTensors grads = model.grads;
                gradoks[0] = CheckTensor(grads.wte, expected_grads.wte, V * C, "dwte");
                gradoks[1] = CheckTensor(grads.wpe, expected_grads.wpe, maxT * C, "dwpe");
                gradoks[2] = CheckTensor(grads.ln1w, expected_grads.ln1w, L * C, "dln1w");
                gradoks[3] = CheckTensor(grads.ln1b, expected_grads.ln1b, L * C, "dln1b");
                gradoks[4] = CheckTensor(grads.qkvw, expected_grads.qkvw, L * 3 * C * C, "dqkvw");
                gradoks[5] = CheckTensor(grads.qkvb, expected_grads.qkvb, L * 3 * C, "dqkvb");
                gradoks[6] = CheckTensor(grads.attprojw, expected_grads.attprojw, L * C * C, "dattprojw");
                gradoks[7] = CheckTensor(grads.attprojb, expected_grads.attprojb, L * C, "dattprojb");
                gradoks[8] = CheckTensor(grads.ln2w, expected_grads.ln2w, L * C, "dln2w");
                gradoks[9] = CheckTensor(grads.ln2b, expected_grads.ln2b, L * C, "dln2b");
                gradoks[10] = CheckTensor(grads.fcw, expected_grads.fcw, L * 4 * C * C, "dfcw");
                gradoks[11] = CheckTensor(grads.fcb, expected_grads.fcb, L * 4 * C, "dfcb");
                gradoks[12] = CheckTensor(grads.fcprojw, expected_grads.fcprojw, L * C * 4 * C, "dfcprojw");
                gradoks[13] = CheckTensor(grads.fcprojb, expected_grads.fcprojb, L * C, "dfcprojb");
                gradoks[14] = CheckTensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw");
                gradoks[15] = CheckTensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb");
                for (int i = 0; i < 16; i++)
                {
                    allOk = allOk && gradoks[i];
                }
            }

            double t4_ms = stopwatch.Elapsed.TotalMilliseconds;
            Update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1, llm);
            double t5_ms = stopwatch.Elapsed.TotalMilliseconds;

            // llm.c did not include Update step when copied although significant phase
            double total_ms = t3_ms + t5_ms - t4_ms;

            losses[step] = model.mean_loss;
            var expectedLoss = expected_losses[step];
            var lossOk = CheckLoss(model.mean_loss, expectedLoss);
            allOk = allOk && lossOk;
            // print the timing information at the end
            Log($"{step,2}: loss {model.mean_loss:F6} exp. {expectedLoss:F6} " +
                $"{(lossOk ? "OK" : "FAIL"),-4} ({total_ms,5:F0} ms = Forward {t1_ms,5:F0} ms ZeroGrad {t2_ms - t1_ms,3:F0} ms Backward {t3_ms - t2_ms,4:F0} ms Update {t5_ms - t4_ms,4:F0} ms)");
        }
        Log($"overall okay: {allOk}");

        llm.Trace(Log);

        // free everything
        free(x);
        free(y);
        free(expected_logits);
        free(expected_loss);
        free(expected_grads_memory);
        Free(&model);
    }

    const float CheckDiffLimit = 0.01f;
    static bool CheckLoss(float a, float b) => Check(a, b);
    static bool Check(float a, float b) => MathF.Abs(a - b) < CheckDiffLimit;

    // poor man's tensor checker
    static unsafe bool CheckTensor(float* actual, float* expected, int n, string label)
    {
        const int printUpTo = 0;//5;
        LogNoNewLine($"{label,-16} ");
        bool ok = true;
        var maxAbsDiff = 0f;
        for (int i = 0; i < n; i++)
        {
            var a = actual[i];
            var e = expected[i];

            var absDiff = MathF.Abs(a - e);
            maxAbsDiff = MathF.Max(absDiff, maxAbsDiff);

            var isOk = absDiff < CheckDiffLimit;
            ok &= isOk;
            if (i < printUpTo)
            {
                Log("");
                LogNoNewLine($"{(isOk ? "OK  " : "FAIL")} {a,15} {e,15} {absDiff,15}");
            }
            if (!isOk) { Debugger.Break(); }
        }
        Log($"TENSOR {(ok ? "OK  " : "FAIL")} MaxAbsDiff {maxAbsDiff,8:F6}");
        return ok;
    }
}
