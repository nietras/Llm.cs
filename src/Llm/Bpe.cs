using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace nietras.LargeLanguageModel;

/// <summary>
/// Simple Byte Pair Encoding for GPT-2. Only the most basic elements.
/// </summary>
// Inspired by and copied from:
//  * https://github.com/microsoft/Tokenizer/blob/main/Tokenizer_C%23/TokenizerLib/Utils/BytePairEncoder.cs
//  * https://github.com/dotnet/machinelearning/blob/main/src/Microsoft.ML.Tokenizers/Model/Tiktoken.cs
// Many unanswered questions regarding special tokens, merges etc.
public partial class Bpe
{
    // Regex patterns based on https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    const string Cl100kBaseRegexPattern = /*lang=regex*/ @"'(?i:[sdmt]|re|ve|ll)|(?>[^\r\n\p{L}\p{N}]?)\p{L}+|\p{N}{1,3}| ?(?>[^\s\p{L}\p{N}]+)[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
    const string P50kBaseRegexPattern = /*lang=regex*/ @"'(?:[sdmt]|re|ve|ll)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    const string O200kBaseRegexPattern = /*lang=regex*/ @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    const string EndOfTextRegexPattern = /*lang=regex*/ @"<\|endoftext\|>";
    [GeneratedRegex(Cl100kBaseRegexPattern)]
    internal static partial Regex Cl100kBaseRegex();
    [GeneratedRegex(P50kBaseRegexPattern)]
    internal static partial Regex P50kBaseRegex();
    [GeneratedRegex(O200kBaseRegexPattern)]
    internal static partial Regex O200kBaseRegex();
    [GeneratedRegex(EndOfTextRegexPattern)]
    internal static partial Regex EndOfTextRegex();

    const string EndOfText = "<|endoftext|>";
    static readonly ReadOnlyMemory<byte> s_endOfTextUtf8 = Encoding.UTF8.GetBytes(EndOfText);
    internal const int Gpt2EndOfTextTokenIndex = 50256;

    //const string FimPrefix = "<|fim_prefix|>";
    //const string FimMiddle = "<|fim_middle|>";
    //const string FimSuffix = "<|fim_suffix|>";
    //const string EndOfPrompt = "<|endofprompt|>";

    readonly IReadOnlyDictionary<ReadOnlyMemory<byte>, int> _bytesToTokenIndex;
    readonly IReadOnlyDictionary<int, ReadOnlyMemory<byte>> _tokenIndexToBytes;
    readonly Regex _specialTokensRegex;
    readonly Regex _splitRegex;

    internal Bpe(IReadOnlyDictionary<ReadOnlyMemory<byte>, int> bytesToTokenIndex,
                 Regex specialTokensRegex, Regex splitRegex)
    {
        _bytesToTokenIndex = bytesToTokenIndex;
        _tokenIndexToBytes = bytesToTokenIndex.ToDictionary(p => p.Value, p => p.Key);
        _specialTokensRegex = specialTokensRegex;
        _splitRegex = splitRegex;
    }

    public static Bpe ReadGpt2FromTiktokenFile(string tikTokenFilePath)
    {
        ArgumentNullException.ThrowIfNull(tikTokenFilePath);
        var bytesToTokenIndex = ReadTikToken(tikTokenFilePath);
        bytesToTokenIndex.Add(s_endOfTextUtf8, Gpt2EndOfTextTokenIndex);
        return new(bytesToTokenIndex, EndOfTextRegex(), P50kBaseRegex());
    }

    public void Encode(ReadOnlySpan<char> text, IList<int> tokenIndices)
    {
        EncodeTokenIndices(text, _specialTokensRegex, _splitRegex, _bytesToTokenIndex, tokenIndices);
    }

    public string? TryDecode(ReadOnlySpan<int> tokenIndices)
    {
        return DecodeTokenIndices(_tokenIndexToBytes, tokenIndices);
    }

    static Dictionary<ReadOnlyMemory<byte>, int> ReadTikToken(string filePath)
    {
        ArgumentNullException.ThrowIfNull(filePath);
        using var reader = new StreamReader(filePath);
        return ReadTikToken(reader);
    }

    static Dictionary<ReadOnlyMemory<byte>, int> ReadTikToken(TextReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);

        var bytesToTokenIndex = new Dictionary<ReadOnlyMemory<byte>, int>(
            ReadOnlyMemoryByteComparer.Instance);
        try
        {
            string? line = null;
            while ((line = reader.ReadLine()) is not null)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                var tokens = line.Split(' ');
                if (tokens.Length != 2)
                {
                    throw new FormatException($"Invalid format in the BPE encoder file stream");
                }

                var tokenBytes = Convert.FromBase64String(tokens[0]);
                var rank = 0;
                if (int.TryParse(tokens[1], out rank))
                {
                    bytesToTokenIndex[tokenBytes] = rank;
                }
                else
                {
                    throw new FormatException($"Can't parse {tokens[1]} to integer");
                }
            }
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load from BPE encoder file stream: {ex.Message}", ex);
        }

        return bytesToTokenIndex;
    }

    static void EncodeTokenIndices(ReadOnlySpan<char> text,
        Regex specialTokensRegex, Regex regex,
        IReadOnlyDictionary<ReadOnlyMemory<byte>, int> bytesToTokenIndex,
        IList<int> tokenIndices)
    {
        var bytes = ArrayPool<byte>.Shared.Rent(2048);
        var specialTokenMatches = specialTokensRegex.EnumerateMatches(text);
        var nextStart = 0;
        while (specialTokenMatches.MoveNext())
        {
            var specialMatch = specialTokenMatches.Current;

            var beforeSpecialToken = text.Slice(nextStart, specialMatch.Index);
            var beforeMatches = regex.EnumerateMatches(beforeSpecialToken);
            foreach (var beforeMatch in beforeMatches)
            {
                var beforeMatchSlice = beforeSpecialToken.Slice(beforeMatch.Index, beforeMatch.Length);
                EncodeMatch(beforeMatchSlice, ref bytes, bytesToTokenIndex, tokenIndices);
            }

            var matchSlice = text.Slice(specialMatch.Index, specialMatch.Length);
            EncodeMatch(matchSlice, ref bytes, bytesToTokenIndex, tokenIndices);
            nextStart = specialMatch.Index + specialMatch.Length;
        }
        var matches = regex.EnumerateMatches(text.Slice(nextStart));
        foreach (var match in matches)
        {
            var matchSlice = text.Slice(nextStart + match.Index, match.Length);
            EncodeMatch(matchSlice, ref bytes, bytesToTokenIndex, tokenIndices);
        }
        ArrayPool<byte>.Shared.Return(bytes);

        static void EncodeMatch(ReadOnlySpan<char> matchSlice,
            ref byte[] bytes,
            IReadOnlyDictionary<ReadOnlyMemory<byte>, int> bytesToTokenIndex,
            IList<int> tokenIndices)
        {
            var maxByteCount = Encoding.UTF8.GetMaxByteCount(matchSlice.Length);
            if (bytes.Length < maxByteCount)
            {
                ArrayPool<byte>.Shared.Return(bytes);
                bytes = ArrayPool<byte>.Shared.Rent(maxByteCount);
            }
            var bytesWritten = Encoding.UTF8.GetBytes(matchSlice, bytes);
            var slice = bytes.AsMemory().Slice(0, bytesWritten);
            EncodeTokenIndices(slice, bytesToTokenIndex, tokenIndices);
        }
    }

    // https://github.com/microsoft/Tokenizer/blob/main/Tokenizer_C%23/TokenizerLib/Utils/BytePairEncoder.cs
    static void EncodeTokenIndices(ReadOnlyMemory<byte> mergingBytes,
        IReadOnlyDictionary<ReadOnlyMemory<byte>, int> bytesToTokenIndex,
        IList<int> tokenIndices)
    {
        ArgumentNullException.ThrowIfNull(tokenIndices);
        if (mergingBytes.Length == 1)
        {
            tokenIndices.Add(bytesToTokenIndex[mergingBytes]);
            return;
        }
        // A direct match should always be the best?
        if (bytesToTokenIndex.TryGetValue(mergingBytes, out var t))
        {
            tokenIndices.Add(t);
            return;
        }

        // TODO: Get rid of List allocation
        var byteIndicesAndRanks = new List<(int Index, int Rank)>();
        for (var i = 0; i < mergingBytes.Length + 1; i++)
        {
            byteIndicesAndRanks.Add((i, int.MaxValue));
        }
        for (var i = 0; i < byteIndicesAndRanks.Count - 2; i++)
        {
            var rank = GetRank(i);
            if (rank != int.MaxValue)
            {
                byteIndicesAndRanks[i] = (byteIndicesAndRanks[i].Index, rank);
            }
        }
        while (byteIndicesAndRanks.Count > 1)
        {
            var minRank = (Index: 0, Rank: int.MaxValue);
            for (var i = 0; i < byteIndicesAndRanks.Count - 1; i++)
            {
                var rank = byteIndicesAndRanks[i].Rank;
                if (rank < minRank.Rank)
                {
                    minRank = (i, rank);
                }
            }
            if (minRank.Rank != int.MaxValue)
            {
                var j = minRank.Index;
                byteIndicesAndRanks[j] = (byteIndicesAndRanks[j].Index, GetRank(j, 1));
                if (j > 0)
                {
                    byteIndicesAndRanks[j - 1] = (byteIndicesAndRanks[j - 1].Index, GetRank(j - 1, 1));
                }
                byteIndicesAndRanks.RemoveAt(j + 1);
            }
            else
            {
                break;
            }
        }
        for (var i = 0; i < byteIndicesAndRanks.Count - 1; i++)
        {
            var sliceStartIndex = byteIndicesAndRanks[i].Index;
            var sliceEndIndex = byteIndicesAndRanks[i + 1].Index;
            var bytes = mergingBytes[sliceStartIndex..sliceEndIndex];
            var tokenIndex = bytesToTokenIndex[bytes];
            tokenIndices.Add(tokenIndex);
        }

        int GetRank(int startIndex, int skip = 0)
        {
            var endIndex = startIndex + skip + 2;
            if (endIndex < byteIndicesAndRanks.Count)
            {
                var sliceStartIndex = byteIndicesAndRanks[startIndex].Index;
                var sliceEndIndex = byteIndicesAndRanks[endIndex].Index;
                var slice = mergingBytes[sliceStartIndex..sliceEndIndex];
                if (bytesToTokenIndex.TryGetValue(slice, out var rank))
                {
                    return rank;
                }
            }
            return int.MaxValue;
        }
    }

    static (int Id, int TokenIndex, int TokenLength)[] Encode(
        ReadOnlyMemory<byte> mergingBytes,
        IReadOnlyDictionary<ReadOnlyMemory<byte>, int> ranks,
        ReadOnlySpan<int> indexMappingSpan)
    {
        ArgumentNullException.ThrowIfNull(ranks);

        if (mergingBytes.Length == 1)
        {
            return [(ranks[mergingBytes], 0, 1)];
        }

        (int Index, int Rank)[]? arrayPoolArray = null;
        var requiredLength = mergingBytes.Length + 1;
        Span<(int Index, int Rank)> byteIndicesAndRanks = requiredLength <= 64 ?
            stackalloc (int, int)[64] :
            (arrayPoolArray = ArrayPool<(int, int)>.Shared.Rent(requiredLength));
        byteIndicesAndRanks = byteIndicesAndRanks.Slice(0, requiredLength);

        for (var i = 0; i < byteIndicesAndRanks.Length; i++)
        {
            byteIndicesAndRanks[i] = (i, int.MaxValue);
        }

        int GetRank(Span<(int Index, int Rank)> byteIndicesAndRanks, int startIndex, int skip = 0)
        {
            var endIndex = startIndex + skip + 2;
            if (endIndex < byteIndicesAndRanks.Length)
            {
                var sliceStart = byteIndicesAndRanks[startIndex].Index;
                var sliceLength = byteIndicesAndRanks[endIndex].Index - sliceStart;
                var slice = mergingBytes.Slice(sliceStart, sliceLength);
                if (ranks.TryGetValue(slice, out var rank))
                {
                    return rank;
                }
            }

            return int.MaxValue;
        }

        for (var i = 0; i < byteIndicesAndRanks.Length - 2; i++)
        {
            var rank = GetRank(byteIndicesAndRanks, i);
            if (rank != int.MaxValue)
            {
                byteIndicesAndRanks[i].Rank = rank;
            }
        }

        while (byteIndicesAndRanks.Length > 1)
        {
            var minRank = (Index: 0, Rank: int.MaxValue);
            for (var i = 0; i < byteIndicesAndRanks.Length - 1; i++)
            {
                if (byteIndicesAndRanks[i].Rank < minRank.Rank)
                {
                    minRank = (i, byteIndicesAndRanks[i].Rank);
                }
            }

            if (minRank.Rank != int.MaxValue)
            {
                var j = minRank.Index;
                byteIndicesAndRanks[j].Rank = GetRank(byteIndicesAndRanks, j, 1);
                if (j > 0)
                {
                    byteIndicesAndRanks[j - 1].Rank = GetRank(byteIndicesAndRanks, j - 1, 1);
                }

                byteIndicesAndRanks.Slice(j + 2).CopyTo(byteIndicesAndRanks.Slice(j + 1));
                byteIndicesAndRanks = byteIndicesAndRanks.Slice(0, byteIndicesAndRanks.Length - 1);
            }
            else
            {
                break;
            }
        }

        var result = new (int Id, int TokenIndex, int TokenLength)[byteIndicesAndRanks.Length - 1];
        for (var i = 0; i < result.Length; i++)
        {
            var startIndex = byteIndicesAndRanks[i].Index;
            var endIndex = byteIndicesAndRanks[i + 1].Index;

            var mappedStartIndex = indexMappingSpan[startIndex];
            var mappedEndIndex = indexMappingSpan[endIndex];

            var finalEndIndex = endIndex;

            if (finalEndIndex > 0 && indexMappingSpan[finalEndIndex - 1] == mappedEndIndex)
            {
                // The partial character/element should be included in the current token.
                finalEndIndex++;
                while (finalEndIndex < indexMappingSpan.Length &&
                       indexMappingSpan[finalEndIndex] == mappedEndIndex)
                {
                    finalEndIndex++;
                }
            }
            var token = ranks[mergingBytes.Slice(startIndex, endIndex - startIndex)];
            result[i] = (token, mappedStartIndex, indexMappingSpan[finalEndIndex] - mappedStartIndex);
        }

        if (arrayPoolArray is not null)
        {
            ArrayPool<(int, int)>.Shared.Return(arrayPoolArray);
        }

        return result;
    }

    static string? DecodeTokenIndices(
        IReadOnlyDictionary<int, ReadOnlyMemory<byte>> tokenIndexToBytes,
        ReadOnlySpan<int> ids)
    {
        // Tiktoken doesn't guarantee a one-to-one correspondence between IDs
        // and UTF-16 words.
        //
        // Consequently, decoding individual IDs into UTF-16 string is not
        // supported; instead, decoding all IDs must be performed collectively.
        // Here's an example case that maps one character to multiple IDs:
        //
        // '⭐' U-2B50 is mapped to IDs [2928, 99834] in the Tiktoken model.
        //
        // In other words, the character '⭐' with UTF-8 code point 0xE2, 0xAD,
        // 0x90 will be mapped by Tiktoken as follows: 0xE2 to [2928] and 0xAD,
        // 0x90 to [99834]. Decoding 2928 and 99834 individually won't
        // reconstruct the original UTF-16 string '⭐' U-2B50; decoding all IDs
        // together is required to get the expected result.

        byte[]? arrayPoolArray = null;
        try
        {
            Span<byte> utf8Bytes = stackalloc byte[256];
            var utf8ByteCount = 0;

            foreach (var id in ids)
            {
                if (tokenIndexToBytes.TryGetValue(id, out ReadOnlyMemory<byte> tokenBytes))
                {
                    if ((uint)utf8ByteCount + (uint)tokenBytes.Length > (uint)utf8Bytes.Length)
                    {
                        ArrayPoolGrow(ref utf8Bytes, ref arrayPoolArray, utf8ByteCount + tokenBytes.Length);
                    }

                    tokenBytes.Span.CopyTo(utf8Bytes.Slice(utf8ByteCount));
                    utf8ByteCount += tokenBytes.Length;
                }
                else
                {
                    return null;
                }
            }

            return Encoding.UTF8.GetString(utf8Bytes.Slice(0, utf8ByteCount));
        }
        finally
        {
            if (arrayPoolArray is not null)
            {
                ArrayPool<byte>.Shared.Return(arrayPoolArray);
            }
        }
    }

    static void ArrayPoolGrow(ref Span<byte> utf8Bytes, ref byte[]? arrayPoolArray, int requiredCapacity)
    {
        var tmp = ArrayPool<byte>.Shared.Rent(Math.Max(utf8Bytes.Length * 2, requiredCapacity));
        utf8Bytes.CopyTo(tmp.AsSpan());
        var toReturn = arrayPoolArray;
        utf8Bytes = arrayPoolArray = tmp;
        if (toReturn is not null)
        {
            ArrayPool<byte>.Shared.Return(toReturn);
        }
    }

    sealed class ReadOnlyMemoryByteComparer : IEqualityComparer<ReadOnlyMemory<byte>>
    {
        public static ReadOnlyMemoryByteComparer Instance { get; } = new();

        ReadOnlyMemoryByteComparer() { }

        public bool Equals(ReadOnlyMemory<byte> x, ReadOnlyMemory<byte> y) =>
            x.Span.SequenceEqual(y.Span);

        public int GetHashCode(ReadOnlyMemory<byte> x)
        {
            var hash = 17;
            foreach (var b in x.Span)
            {
                hash = hash * 31 + b;
            }
            return hash;
        }
    }
}
