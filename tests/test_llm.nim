## Strict reply parsing: what `extractJsonObject`, `parseDecision`, and
## `parseScriptKind` accept and reject. No action compatibility shims, no
## legacy aliases, no guessing from prose or card letters.

import std/[json, unittest]
import gozu/[llm, sim]

proc parseReply(raw: string, mode = mGoofspiel): Decision =
  parseDecision(extractJsonObject(raw), mode)

suite "strict JSON extraction":
  test "accepts a single bare JSON object":
    let node = extractJsonObject("""  {"bid": 9, "say": "going high", "notes": "keep 13"}  """)
    check node["bid"].getInt() == 9
    check node["say"].getStr() == "going high"
    check node["notes"].getStr() == "keep 13"

  test "accepts a single ordinary outer Markdown code fence (json or plain)":
    let tagged = extractJsonObject("```json\n{\"bid\": 7}\n```")
    check tagged["bid"].getInt() == 7
    let taggedUpper = extractJsonObject("  ```JSON\n{\"bid\": 8, \"say\": \"hi\"}\n```  ")
    check taggedUpper["bid"].getInt() == 8
    let plain = extractJsonObject("```\n{\"bid\": 5, \"notes\": \"save\"}\n```")
    check plain["bid"].getInt() == 5

  test "rejects surrounding unrelated prose":
    expect GozuError:
      discard extractJsonObject("I will bid 9:\n{\"bid\": 9}")
    expect GozuError:
      discard extractJsonObject("{\"bid\": 9}\nGood luck!")
    expect GozuError:
      discard extractJsonObject("Here is my move:\n```json\n{\"bid\": 9}\n```")
    expect GozuError:
      discard extractJsonObject("```json\n{\"bid\": 9}\n```\nHope that works.")

  test "rejects extra objects and multiple fences":
    expect GozuError:
      discard extractJsonObject("{\"bid\": 9} {\"bid\": 10}")
    expect GozuError:
      discard extractJsonObject("```json\n{\"bid\": 9}\n```\n```json\n{\"bid\": 10}\n```")

  test "rejects malformed fences and non-object JSON":
    expect GozuError:
      discard extractJsonObject("```python\n{\"bid\": 9}\n```")
    expect GozuError:
      discard extractJsonObject("```json\n{\"bid\": 9}```")
    expect GozuError:
      discard extractJsonObject("````json\n{\"bid\": 9}\n````")
    expect GozuError:
      discard extractJsonObject("[9]")
    expect GozuError:
      discard extractJsonObject("9")
    expect GozuError:
      discard extractJsonObject("")

suite "strict bid parsing":
  test "accepts integer bid and optional say/notes":
    let minimal = parseReply("""{"bid": 9}""")
    check minimal.bid == 9
    check minimal.say == ""
    check minimal.notes == ""

    let withNulls = parseReply("""{"bid": 3, "say": null, "notes": null}""", mOshiZumo)
    check withNulls.bid == 3
    check withNulls.say == ""
    check withNulls.notes == ""

    let full = parseReply("```json\n{\"bid\": 11, \"say\": \"bold\", \"notes\": \"track 13\"}\n```")
    check full.bid == 11
    check full.say == "bold"
    check full.notes == "track 13"

  test "rejects string bids, floats, card letters, and prose":
    for bad in [
      """{"bid": "11"}""",
      """{"bid": "  13  "}""",
      """{"bid": "11 — the king"}""",
      """{"bid": "7, keeping the ace back"}""",
      """{"bid": "A"}""",
      """{"bid": "K"}""",
      """{"bid": "K."}""",
      """{"bid": "a bid of 11"}""",
      """{"bid": 7.6}""",
      """{"bid": 7.0}""",
      """{"bid": true}""",
      """{"bid": null}""",
      """{"say": "hello"}"""
    ]:
      expect GozuError:
        discard parseReply(bad)

  test "rejects wrong types on optional fields and unexpected legacy aliases":
    expect GozuError:
      discard parseReply("""{"bid": 5, "say": 123}""")
    expect GozuError:
      discard parseReply("""{"bid": 5, "notes": ["a"]}""")
    expect GozuError:
      discard parseReply("""{"bid": 5, "card": 5}""")
    expect GozuError:
      discard parseReply("""{"card": 5}""")

suite "strict scripted baseline parsing":
  test "accepts only canonical names":
    check parseScriptKind("") == skNone
    check parseScriptKind("none") == skNone
    check parseScriptKind("match") == skMatch
    check parseScriptKind("  HOARD  ") == skHoard

  test "rejects legacy aliases and unknown values":
    for legacy in ["1", "true", "yes", "hoarder", "random"]:
      expect GozuError:
        discard parseScriptKind(legacy)
