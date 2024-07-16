using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace nietras.LargeLanguageModel.Test;

[TestClass]
public partial class BpeTest
{
    const string Gpt2TiktokenFilePath = @"../../../../artifacts/gpt2.tiktoken";
    const string EndOfText = "<|endoftext|>";
    readonly Bpe _sut;
    readonly List<int> _actualTokenIndices = [];

    public BpeTest()
    {
        _sut = Bpe.ReadGpt2FromTiktokenFile(Gpt2TiktokenFilePath);
    }

    [TestMethod]
    public void BpeTest_Gpt2_librs_txt()
    {
        var text = ReadAndSanitizeFile("lib.rs.txt");

        var json = File.ReadAllText("lib.rs.tokens_gpt2.json");
        IReadOnlyList<int> expected = JsonSerializer.Deserialize<int[]>(json)!;

        AssertEncodeDecode(text, expected);
    }

    [TestMethod]
    public void BpeTest_Gpt2_Simple()
    {
        AssertEncodeDecode("", []);

        AssertEncodeDecode("\0", [188]);

        AssertEncodeDecode("The quick brown fox jumps over the lazy dog!",
                           [464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 0]);

        AssertEncodeDecode($"The quick brown fox {EndOfText} jumps over the lazy dog!",
                           [464, 2068, 7586, 21831, 220, 50256, 18045, 625, 262, 16931, 3290, 0]);

        AssertEncodeDecode("hello world", [31373, 995]);
        AssertEncodeDecode("hello <|endoftext|>", [31373, 220, 50256]);
    }

    [TestMethod]
    public void BpeTest_Gpt2_SimpleRepeated()
    {
        AssertEncodeDecode("0", [15]);
        AssertEncodeDecode("00", [405]);
        AssertEncodeDecode("000", [830]);
        AssertEncodeDecode("0000", [2388]);
        AssertEncodeDecode("00000", [20483]);
        AssertEncodeDecode("000000", [10535]);
        AssertEncodeDecode("0000000", [24598]);
        AssertEncodeDecode("00000000", [8269]);
        AssertEncodeDecode("000000000", [10535, 830]);
        AssertEncodeDecode("0000000000", [8269, 405]);
        AssertEncodeDecode("00000000000", [8269, 830]);
        AssertEncodeDecode("000000000000", [8269, 2388]);
        AssertEncodeDecode("0000000000000", [8269, 20483]);
        AssertEncodeDecode("00000000000000", [8269, 10535]);
        AssertEncodeDecode("000000000000000", [8269, 24598]);
        AssertEncodeDecode("0000000000000000", [25645]);
        AssertEncodeDecode("00000000000000000", [8269, 10535, 830]);
    }

    void AssertEncodeDecode(string text, IReadOnlyList<int> expected)
    {
        AssertEncode(text, expected);
        AssertDecode(_actualTokenIndices, text);
    }

    void AssertEncode(string text, IReadOnlyList<int> expected)
    {
        _actualTokenIndices.Clear();
        _sut.Encode(text, _actualTokenIndices);
        Assert.AreEqual(expected.Count, _actualTokenIndices.Count);
        CollectionAssert.AreEqual((ICollection)expected, _actualTokenIndices);
    }

    void AssertDecode(List<int> tokenIndices, string expected)
    {
        var actual = _sut.TryDecode(CollectionsMarshal.AsSpan(tokenIndices));
        Assert.AreEqual(expected, actual);
    }

    // Test running copy the test data files to the output folder but sometimes
    // the file content is mutated replacing '\n' with '\r\n'. This method reads
    // the file and removes the extra inserted '\r' characters. Having '\r' in
    // the file content will cause the tests to fail.
    static string ReadAndSanitizeFile(string path)
    {
        var text = File.ReadAllText(path);
        return text.ReplaceLineEndings("\n");
    }
}
