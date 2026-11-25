## Hosted calls must reach the native sidecar without provider credentials.
include "../src/gozu/llm"

block:
  putEnv("COWORLD_LLM_ENDPOINT", "http://127.0.0.1:9100/")
  putEnv("COWORLD_LLM_MODEL", "anthropic/claude-sonnet-4.6")
  putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://retired.invalid")
  putEnv("ANTHROPIC_API_KEY", "local-key-must-not-be-used")
  let client = newLlmClient(GameConfig())
  for slot in 0 .. 1:
    let request = client.requestFor("rules", "private view", slot)
    doAssert request.url == "http://127.0.0.1:9100/v1/messages"
    doAssert request.headers["X-Coworld-Player-Slot"] == $slot
    let body = parseJson(request.body)
    doAssert body["model"].getStr() == "anthropic/claude-sonnet-4.6"
    doAssert not body.hasKey("anthropic_version")
    doAssert not body.hasKey("output_config")
  echo "hosted sidecar routing and seat attribution passed"

block:
  delEnv("COWORLD_LLM_ENDPOINT")
  delEnv("COWORLD_LLM_MODEL")
  delEnv("BEDROCK_MODEL")
  delEnv("ANTHROPIC_API_KEY")
  putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://127.0.0.1:9200/")
  putEnv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
  let client = newLlmClient(defaultGameConfig())
  doAssert client.transport == ltBedrock
  doAssert client.model == DefaultBedrockModel
  let request = client.requestFor("rules", "private view", 0)
  doAssert request.url == "http://127.0.0.1:9200/model/" & DefaultBedrockModel & "/invoke"

  var throttled = false
  try:
    discard client.textOf(Response(code: 429, body: "throttled"), "", request.url)
  except LlmInfraError:
    throttled = true
  doAssert throttled
  doAssert client.model == DefaultBedrockModel
  doAssert not client.disabled

  var denied = false
  try:
    discard client.textOf(Response(code: 403, body: "Model access is denied"), "", request.url)
  except LlmInfraError:
    denied = true
  doAssert denied
  doAssert client.model == DefaultBedrockModel
  doAssert client.disabled

  putEnv("BEDROCK_MODEL", "us.anthropic.claude-pinned-test")
  let pinned = newLlmClient(defaultGameConfig())
  doAssert pinned.model == "us.anthropic.claude-pinned-test"
  doAssert pinned.requestFor("rules", "private view", 0).url ==
    "http://127.0.0.1:9200/model/us.anthropic.claude-pinned-test/invoke"
  echo "bedrock single-model routing and no-fallback error handling passed"
