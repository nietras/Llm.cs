using System.Diagnostics;
using System.IO;
using System.Reflection;

namespace nietras.LargeLanguageModel;

public static partial class Gpt2
{
    // ----------------------------------------------------------------------------
    // main training loop
    public static unsafe void Train()
    {
        var location = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        var dataDirectory = Path.Combine(location!, "../../../");
        // build the GPT-2 model from a checkpoint
        GPT2 model;
        gpt2_build_from_checkpoint(&model, dataDirectory + "gpt2_124M.bin");

        // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
        var tiny_stories_train = dataDirectory + "TinyStories_train.bin";
        var tiny_stories_val = dataDirectory + "TinyStories_val.bin";
        var tiny_shakespeare_train = dataDirectory + "tiny_shakespeare_train.bin";
        var tiny_shakespeare_val = dataDirectory + "tiny_shakespeare_val.bin";
        var train_tokens = File.Exists(tiny_shakespeare_train) ? tiny_shakespeare_train : tiny_stories_train;
        var val_tokens = File.Exists(tiny_shakespeare_val) ? tiny_shakespeare_val : tiny_stories_val;
        int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
        int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
        using DataLoader train_loader = new(train_tokens, B, T);
        Log($"train dataset num_batches: {train_loader.num_batches}");

        using DataLoader val_loader = new(val_tokens, B, T);
        Log($"val dataset num_batches: {val_loader.num_batches}");
        int val_num_batches = 10;

        // some memory for generating samples from the model
        ulong rng_state = 1337;
        // during inference step we'll generate sequences of this many tokens
        const int gen_max_length = 64;
        int* gen_tokens = stackalloc int[gen_max_length];

        // train
        var stopwatch = new Stopwatch();
        for (int step = 0; step <= 20; step++)
        {

            // once in a while estimate the validation loss
            if (step % 10 == 0)
            {
                float val_loss = 0.0f;
                val_loader.dataloader_reset();
                for (int i = 0; i < val_num_batches; i++)
                {
                    val_loader.dataloader_next_batch();
                    gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                    val_loss += model.mean_loss;
                }
                val_loss /= val_num_batches;
                Log($"val loss {val_loss}");
            }

            // once in a while do model inference to print generated text
            if (step > 0 && step % 20 == 0)
            {
                gen_tokens[0] = GPT2_EOT; // the GPT-2 EOT token kicks off the generation
                for (int t = 1; t < gen_max_length; t++)
                {
                    // note that inference is wasteful here because
                    // for each t, we re-compute all activations between 0 and t
                    // leaving this alone because you want separate code for inference anyway
                    // the inference here is just for sanity checking purposes
                    gpt2_forward(&model, gen_tokens, null, 1, t);
                    float* probs = model.acts.probs + (t - 1) * model.config.vocab_size;
                    float coin = random_f32(&rng_state);
                    int next_token = sample_mult(probs, model.config.vocab_size, coin);
                    gen_tokens[t] = next_token;
                }
                Log("generated: ");
                for (int t = 0; t < gen_max_length; t++)
                {
                    Log($"{gen_tokens[t]} ");
                }
                Log("");
            }

            // do a training step
            stopwatch.Restart();
            train_loader.dataloader_next_batch();
            gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
            gpt2_zero_grad(&model);
            gpt2_backward(&model);
            gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
            double time_elapsed_ms = stopwatch.Elapsed.TotalMilliseconds;
            Log($"step {step}: train loss {model.mean_loss} (took {time_elapsed_ms} ms)");
        }

        // free
        train_loader.dataloader_free();
        val_loader.dataloader_free();
        gpt2_free(&model);
    }
}
