using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

namespace nietras.LargeLanguageModel;

internal static partial class Gpt2
{
    public static unsafe void Test()
    {
        var location = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        var dataDirectory = Path.Combine(location!, "../../../");
        // build the GPT-2 model from a checkpoint
        GPT2 model;
        gpt2_build_from_checkpoint(&model, dataDirectory + "gpt2_124M.bin");

        int C = model.config.channels;
        int V = model.config.vocab_size;
        int maxT = model.config.max_seq_len;
        int L = model.config.num_layers;

        using var state_file = File.OpenRead(dataDirectory + "gpt2_124M_debug_state.bin");
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
        float* expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes);

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

        // overall OK signal for the test
        bool allok = true;

        // let's do 10 training iterations, following the pytorch code
        float* losses = stackalloc float[10];
        var stopwatch = new Stopwatch();
        for (int step = 0; step < 10; step++)
        {
            stopwatch.Restart();

            gpt2_forward(&model, x, y, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);

            double time_elapsed_s = stopwatch.Elapsed.TotalSeconds;

            if (step == 0)
            {
                // error checking at step 0 for reference activations/gradients

                // at this point, target should be equal to expected_logits, let's compare
                bool logits_ok = true;
                for (int i = 0; i < B * T * V; i++)
                {
                    if (i < 3)
                    {
                        Log($"{expected_logits[i]} {model.acts.logits[i]}");
                    }
                    if (MathF.Abs(expected_logits[i] - model.acts.logits[i]) >= 1e-2)
                    {
                        Log($"MISMATCH AT INDEX {i}: {expected_logits[i]} {model.acts.logits[i]}");
                        logits_ok = false;
                        break;
                    }
                }
                if (!logits_ok) { Log("NOT "); }
                Log("OK (LOGITS)");
                allok = allok && logits_ok;

                // compare the achieved loss
                if (MathF.Abs(model.mean_loss - *expected_loss) >= 1e-2)
                {
                    Log($"LOSS MISMATCH: {model.mean_loss} {*expected_loss}");
                    allok = false;
                }
                else
                {
                    Log($"LOSS OK: {model.mean_loss} {*expected_loss}");
                }

                // finally check all the gradients
                var gradoks = new bool[16];
                ParameterTensors grads = model.grads;
                gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V * C, "dwte");
                gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT * C, "dwpe");
                gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L * C, "dln1w");
                gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L * C, "dln1b");
                gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L * 3 * C * C, "dqkvw");
                gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L * 3 * C, "dqkvb");
                gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L * C * C, "dattprojw");
                gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L * C, "dattprojb");
                gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L * C, "dln2w");
                gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L * C, "dln2b");
                gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L * 4 * C * C, "dfcw");
                gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L * 4 * C, "dfcb");
                gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L * C * 4 * C, "dfcprojw");
                gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L * C, "dfcprojb");
                gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw");
                gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb");
                for (int i = 0; i < 16; i++)
                {
                    allok = allok && gradoks[i];
                }
            }

            gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);

            // print the timing information at the end
            Log($"step {step}: loss {model.mean_loss} (took {time_elapsed_s * 1000} ms)");
            losses[step] = model.mean_loss;
        }

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
        // compare
        for (int i = 0; i < 10; i++)
        {
            if (MathF.Abs(losses[i] - expected_losses[i]) >= 1e-2)
            {
                Log($"LOSS MISMATCH AT STEP {i}: {losses[i]} {expected_losses[i]}");
                allok = false;
            }
            else
            {
                Log($"loss ok at step {i}: {losses[i]} {expected_losses[i]}");
            }
        }

        Log($"overall okay: {allok}");

        // free everything
        free(x);
        free(y);
        free(expected_logits);
        free(expected_loss);
        free(expected_grads_memory);
        gpt2_free(&model);
    }

    // poor man's tensor checker
    static unsafe bool check_tensor(float* a, float* b, int n, string label)
    {
        int print_upto = 5;
        bool ok = true;
        Log($"{label}");

        for (int i = 0; i < n; i++)
        {
            if (MathF.Abs(a[i] - b[i]) <= 1e-2)
            {
                if (i < print_upto) { Log("OK "); }
            }
            else
            {
                if (i < print_upto) { Log("NOT OK "); }
                ok = false;
            }
            if (i < print_upto) { Log($"{a[i]} {b[i]}"); }
        }
        // print the final result
        if (ok)
        {
            Log("TENSOR OK");
        }
        else
        {
            Log("TENSOR NOT OK");
        }
        return ok;
    }
}
