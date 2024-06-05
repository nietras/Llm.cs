namespace nietras.LargeLanguageModel;

public interface ILlm
{
    /// <summary>
    /// Forward pass of the embedding layer.
    /// </summary>
    /// <param name="tokenIndices">Pointer to the input tensor of token indices/ids.</param>
    /// <param name="tokenEmbeddings">Pointer to the token embeddings tensor.</param>
    /// <param name="positionEmbeddings">Pointer to the position embeddings tensor.</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="channelCount">The number of channels.</param>
    /// <param name="output">Pointer to the output tensor.</param>
    static abstract unsafe void EmbedForward(
            int* tokenIndices, float* tokenEmbeddings, float* positionEmbeddings,
            int batchSize, int tokenCount, int channelCount,
            float* output);
    /// <summary>
    /// Backward pass of the embedding layer.
    /// </summary>
    /// <param name="δoutput">Pointer to the output derivative tensor of shape [batchSize, tokenCount, channelCount].</param>
    /// <param name="tokenIndices">Pointer to the token indices tensor of shape [batchSize, tokenCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="channelCount">The number of channels.</param>
    /// <param name="δtokenEmbeddings">Pointer to the token embeddings derivative tensor of shape [vocabularySize, channelCount].</param>
    /// <param name="δpositionEmbeddings">Pointer to the position embeddings derivative tensor of shape [maxTokenCount, channelCount].</param>
    static abstract unsafe void EmbedBackward(
            float* δoutput, int* tokenIndices,
            int batchSize, int tokenCount, int channelCount,
            float* δtokenEmbeddings, float* δpositionEmbeddings);

    /// <summary>
    /// Forward pass of Layer Normalization.
    /// </summary>
    /// <param name="input">The input tensor of shape [batchSize, tokenCount, channelCount].</param>
    /// <param name="weight">The weight tensor of shape [channelCount].</param>
    /// <param name="bias">The bias tensor of shape [channelCount].</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="tokenCount">The token count.</param>
    /// <param name="channelCount">The channel count.</param>
    /// <param name="mean">The mean tensor of shape [batchSize, tokenCount].</param>
    /// <param name="invStdDev">The inverse standard deviation tensor of shape [batchSize, tokenCount].</param>
    /// <param name="output">The output tensor of shape [batchSize, tokenCount, channelCount].</param>
    static abstract unsafe void LayerNormForward(
            float* input, float* weight, float* bias, int batchSize, int tokenCount, int channelCount,
            float* mean, float* invStdDev, float* output);
    /// <summary>
    /// Backward pass of Layer Normalization.
    /// </summary>
    /// <param name="δoutput">The gradients of the output tensor. Shape: [batchSize, tokenCount, channelCount].</param>
    /// <param name="input">The input tensor. Shape: [batchSize, tokenCount, channelCount].</param>
    /// <param name="weight">The weight tensor. Shape: [channelCount].</param>
    /// <param name="mean">The mean tensor. Shape: [batchSize, tokenCount].</param>
    /// <param name="invStdDev">The inverse standard deviation tensor. Shape: [batchSize, tokenCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="channelCount">The number of channels.</param>
    /// <param name="δweight">The gradients of the weight tensor. Shape: [channelCount].</param>
    /// <param name="δbias">The gradients of the bias tensor. Shape: [channelCount].</param>
    /// <param name="δinput">The gradients of the input tensor. Shape: [batchSize, tokenCount, channelCount].</param>
    static abstract unsafe void LayerNormBackward(
            float* δoutput, float* input, float* weight, float* mean, float* invStdDev,
            int batchSize, int tokenCount, int channelCount,
            float* δweight, float* δbias, float* δinput);

    /// <summary>
    /// Matrix multiplication between the input tensor and transposed weight tensor 
    /// and optionally adds bias if not null.
    /// </summary>
    /// <remarks>
    /// Note how this is more of a vector * matrix operation than a matrix * matrix operation, 
    /// since for each batch and token it multiples input row vector with 
    /// weight (transposed) matrix and adds bias vector.
    /// </remarks>
    /// <param name="input">The input tensor of shape [batchSize, tokenCount, inputChannelCount].</param>
    /// <param name="weight">The weight tensor (transposed) of shape [outputChannelCount, inputChannelCount].</param>
    /// <param name="bias">The bias tensor of shape [outputChannelCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="inputChannelCount">The number of input channels.</param>
    /// <param name="outputChannelCount">The number of output channels.</param>
    /// <param name="output">The output tensor of shape [batchSize, tokenCount, outputChannelCount].</param>
    static abstract unsafe void MatMulForward(
            float* input, float* weight, float* bias,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* output);
    /// <summary>
    /// Backward pass of matrix multiplication operation.
    /// </summary>
    /// <param name="δoutput">The gradient of the output tensor. Shape: [batchSize, tokenCount, outputChannelCount].</param>
    /// <param name="input">The input tensor. Shape: [batchSize, tokenCount, inputChannelCount].</param>
    /// <param name="weight">The weight tensor. Shape: [outputChannelCount, inputChannelCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="inputChannelCount">The number of input channels.</param>
    /// <param name="outputChannelCount">The number of output channels.</param>
    /// <param name="δweight">The gradient of the weight tensor. Shape: [outputChannelCount, inputChannelCount].</param>
    /// <param name="δbias">The gradient of the bias tensor. Shape: [outputChannelCount].</param>
    /// <param name="δinput">The gradient of the input tensor. Shape: [batchSize, tokenCount, inputChannelCount].</param>
    static abstract unsafe void MatMulBackward(
            float* δoutput, float* input, float* weight,
            int batchSize, int tokenCount, int inputChannelCount, int outputChannelCount,
            float* δweight, float* δbias, float* δinput);

    /// <summary>
    /// Forward pass of the Attention layer.
    /// </summary>
    /// <param name="input">The input tensor of shape [batchSize, tokenCount, 3 * channelCount] holding the query, key, and value (Q, K, V) vectors.</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens in the sequence.</param>
    /// <param name="channelCount">The number of channels in the input tensor.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="preAttention">The pre-attention tensor of shape [batchSize, headCount, tokenCount, tokenCount] that holds the pre-attention scores.</param>
    /// <param name="postAttention">The post-attention tensor of shape [batchSize, headCount, tokenCount, tokenCount] that holds the post-attention scores.</param>
    /// <param name="output">The output tensor of shape [batchSize, tokenCount, channelCount] that holds the attention output.</param>
    static abstract unsafe void AttentionForward(
        float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* preAttention, float* postAttention, float* output);
    /// <summary>
    /// Backward pass of the Attention layer.
    /// </summary>
    /// <param name="δoutput">The gradient of the output tensor. Shape: [batchSize, tokenCount, channelCount].</param>
    /// <param name="postAttention">The post-softmax attention tensor. Shape: [batchSize, headCount, tokenCount, tokenCount].</param>
    /// <param name="input">The input tensor. Shape: [batchSize, tokenCount, 3 * channelCount (Q, K, V)].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="channelCount">The number of channels.</param>
    /// <param name="headCount">The number of attention heads.</param>
    /// <param name="δpreAttention">The gradient of the pre-softmax attention tensor. Shape: [batchSize, headCount, tokenCount, tokenCount].</param>
    /// <param name="δpostAttention">The gradient of the attention tensor. Shape: [batchSize, headCount, tokenCount, tokenCount].</param>
    /// <param name="δinput">The gradient of the input tensor. Shape: [batchSize, tokenCount, 3 * channelCount (Q, K, V)].</param>
    static abstract unsafe void AttentionBackward(
        float* δoutput, float* postAttention, float* input,
        int batchSize, int tokenCount, int channelCount, int headCount,
        float* δpreAttention, float* δpostAttention, float* δinput);

    /// <summary>
    /// Forward pass of Tanh based approximate GeLU (Gaussian Error Linear Unit).
    /// </summary>
    /// <param name="input">The input array.</param>
    /// <param name="count">The number of elements in the <paramref name="input"/> and <paramref name="output"/> array.</param>
    /// <param name="output">The output array.</param>
    static abstract unsafe void GeLUForward(float* input, int count, float* output);
    /// <summary>
    /// Backward pass of Tanh based approximate GeLU (Gaussian Error Linear Unit).
    /// </summary>
    /// <param name="δoutput">The gradient of the output.</param>
    /// <param name="input">The input values.</param>
    /// <param name="count">The number of elements in the <paramref name="δoutput"/>, <paramref name="input"/> and <paramref name="δinput"/>.</param>
    /// <param name="δinput">The gradient of the input.</param>
    static abstract unsafe void GeLUBackward(float* δoutput, float* input, int count, float* δinput);

    /// <summary>
    /// Forward pass of the residual/add operation.
    /// </summary>
    /// <param name="left">The input array for the left operand.</param>
    /// <param name="right">The input array for the right operand.</param>
    /// <param name="count">The number of elements in the arrays.</param>
    /// <param name="output">The output array.</param>
    static abstract unsafe void ResidualForward(float* left, float* right, int count, float* output);
    /// <summary>
    /// Backward pass of the residual/add connection.
    /// </summary>
    /// <param name="δoutput">The gradients of the output.</param>
    /// <param name="count">The number of elements in the tensors.</param>
    /// <param name="δleft">The gradients of the left tensor.</param>
    /// <param name="δright">The gradients of the right tensor.</param>
    static abstract unsafe void ResidualBackward(float* δoutput, int count, float* δleft, float* δright);

    /// <summary>
    /// Forward pass of the softmax layer.
    /// </summary>
    /// <param name="logits">The input logits. Shape: [batchSize, tokenCount, vocabularySize]</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="vocabularySize">The size of the vocabulary.</param>
    /// <param name="probabilities">The output probabilities. Sums to 1.0 for each batch,token. Shape: [batchSize, tokenCount, vocabularySize]</param>
    static abstract unsafe void SoftmaxForward(float* logits,
        int batchSize, int tokenCount, int vocabularySize,
        float* probabilities);
    /// <summary>
    /// Forward pass of the cross-entropy loss.
    /// </summary>
    /// <param name="probabilities">The input probabilities. Shape: [batchSize, tokenCount, vocabularySize]</param>
    /// <param name="targetTokenIndices">The target token indices. Shape: [batchSize, tokenCount]</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="vocabularySize">The size of the vocabulary.</param>
    /// <param name="losses">The output losses. Shape: [batchSize, tokenCount]</param>
    static abstract unsafe void CrossEntropyForward(
        float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* losses);
    /// <summary>
    /// Backward pass of both CrossEntropy and Softmax.
    /// </summary>
    /// <param name="δlosses">The gradients of the losses with respect to the output probabilities. Shape: [batchSize, tokenCount].</param>
    /// <param name="probabilities">The output probabilities. Shape: [batchSize, tokenCount, vocabularySize].</param>
    /// <param name="targetTokenIndices">The indices of the target tokens. Shape: [batchSize, tokenCount].</param>
    /// <param name="batchSize">The size of the batch.</param>
    /// <param name="tokenCount">The number of tokens.</param>
    /// <param name="vocabularySize">The size of the vocabulary.</param>
    /// <param name="δlogits">The gradients of the logits with respect to the input. Shape: [batchSize, tokenCount, vocabularySize].</param>
    static abstract unsafe void CrossEntropySoftmaxBackward(
        float* δlosses, float* probabilities, int* targetTokenIndices,
        int batchSize, int tokenCount, int vocabularySize,
        float* δlogits);
}
