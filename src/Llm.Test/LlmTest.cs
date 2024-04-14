using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace nietras.LargeLanguageModel.Test;

[TestClass]
public class LlmTest
{
    readonly IReadOnlyList<char> _supportedLlmarators = LlmDefaults
        .AutoDetectLlmarators.Concat([' ', '~']).ToArray();

    [TestMethod]
    public void LlmTest_Ctor_Empty()
    {
        var sep = new Llm();
        Assert.AreEqual(LlmDefaults.Llmarator, sep.Llmarator);
    }

    [TestMethod]
    public void LlmTest_Ctor()
    {
        foreach (var separator in _supportedLlmarators)
        {
            var sep = new Llm(separator);
            Assert.AreEqual(separator, sep.Llmarator);
        }
    }

    [TestMethod]
    public void LlmTest_New()
    {
        foreach (var separator in _supportedLlmarators)
        {
            var sep = Llm.New(separator);
            Assert.AreEqual(separator, sep.Llmarator);
        }
    }

    [TestMethod]
    public void LlmTest_Auto()
    {
        var maybeLlm = Llm.Auto;
        Assert.IsNull(maybeLlm);
    }

    [TestMethod]
    public void LlmTest_Equality()
    {
        var x1 = new Llm(';');
        var x2 = new Llm(';');
        var other = new Llm(',');

        Assert.IsTrue(x1 == x2);
        Assert.IsTrue(x2 == x1);
        Assert.IsFalse(x1 == other);
        Assert.IsTrue(x1 != other);

        Assert.IsTrue(x1.Equals(x2));
        Assert.IsTrue(x2.Equals(x1));
        Assert.IsFalse(x1.Equals(other));
    }

    [TestMethod]
    public void LlmTest_Llmarator_LessThanMin_Throws()
    {
        var separator = (char)(Llm.Min.Llmarator - 1);
        var expectedMessage = "'\u001f':31 is not supported. Must be inside [32..126]. (Parameter 'separator')";
        AssertLlmaratorThrows<ArgumentOutOfRangeException>(separator, expectedMessage);
    }

    [TestMethod]
    public void LlmTest_Llmarator_GreaterThanMax_Throws()
    {
        var separator = (char)(Llm.Max.Llmarator + 1);
        var expectedMessage = "'\u007f':127 is not supported. Must be inside [32..126]. (Parameter 'separator')";
        AssertLlmaratorThrows<ArgumentOutOfRangeException>(separator, expectedMessage);
    }

    [TestMethod]
    public void LlmTest_Llmarator_LineFeed_Throws()
    {
        var separator = '\n';
        var expectedMessage = "'\n':10 is not supported. Must be inside [32..126]. (Parameter 'separator')";
        AssertLlmaratorThrows<ArgumentOutOfRangeException>(separator, expectedMessage);
    }

    [TestMethod]
    public void LlmTest_Llmarator_CarriageReturn_Throws()
    {
        var separator = '\r';
        var expectedMessage = "'\r':13 is not supported. Must be inside [32..126]. (Parameter 'separator')";
        AssertLlmaratorThrows<ArgumentOutOfRangeException>(separator, expectedMessage);
    }

    [TestMethod]
    public void LlmTest_Llmarator_Quote_Throws()
    {
        var separator = '\"';
        var expectedMessage = "'\"':34 is not supported. (Parameter 'separator')";
        AssertLlmaratorThrows<ArgumentException>(separator, expectedMessage);
    }

    [TestMethod]
    public void LlmTest_Llmarator_Comment_Throws()
    {
        var separator = '#';
        var expectedMessage = "'#':35 is not supported. (Parameter 'separator')";
        AssertLlmaratorThrows<ArgumentException>(separator, expectedMessage);
    }

    static void AssertLlmaratorThrows<TException>(char separator, string expectedMessage)
        where TException : Exception
    {
        Action[] actions =
        [
            () => { var s = new Llm(separator); },
            () => { var s = Llm.New(separator); },
            () => { var s = Llm.Default with { Llmarator = separator }; },
        ];
        foreach (var action in actions)
        {
            var e = Assert.ThrowsException<TException>(action);
            Assert.AreEqual(expectedMessage, e.Message);
        }
    }
}
