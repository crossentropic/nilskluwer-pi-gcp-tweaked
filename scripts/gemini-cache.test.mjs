import assert from "node:assert/strict";
import {
  clearGoogleVertexClientCache,
  getOrCreateGeminiVertexClient,
  getGoogleVertexModels,
  DEFAULT_GEMINI_CACHE_TTL_SECONDS,
  convertMessages,
  extractToolResultText,
  isGemini3OrNewer,
  mapThinkingLevel,
  buildParams,
} from "../extensions/gcp-agent-platform/gemini.ts";

async function testClientMemoization() {
  clearGoogleVertexClientCache();

  // 1. Calling with identical parameters should return the exact same instance
  const client1 = await getOrCreateGeminiVertexClient("proj-a", "eu", "/path/to/key1.json");
  const client2 = await getOrCreateGeminiVertexClient("proj-a", "eu", "/path/to/key1.json");
  assert.strictEqual(client1, client2, "Clients with identical profile must be the same memoized instance");

  // 2. Calling with different project or location must return a different instance
  const clientDiffProj = await getOrCreateGeminiVertexClient("proj-b", "eu", "/path/to/key1.json");
  assert.notStrictEqual(client1, clientDiffProj, "Clients with different projects must not share instances");

  const clientDiffLoc = await getOrCreateGeminiVertexClient("proj-a", "us-central1", "/path/to/key1.json");
  assert.notStrictEqual(client1, clientDiffLoc, "Clients with different locations must not share instances");

  const clientDiffKey = await getOrCreateGeminiVertexClient("proj-a", "eu", "/path/to/key2.json");
  assert.notStrictEqual(client1, clientDiffKey, "Clients with different key files must not share instances");

  // 3. One-off custom headers should bypass the shared cache to prevent header pollution
  const clientWithHeaders = await getOrCreateGeminiVertexClient("proj-a", "eu", "/path/to/key1.json", {
    "x-session-id": "transient-123",
  });
  assert.notStrictEqual(client1, clientWithHeaders, "Custom headers must produce a dedicated unshared client");

  // 4. clearGoogleVertexClientCache wipes the cache
  clearGoogleVertexClientCache();
  const clientAfterClear = await getOrCreateGeminiVertexClient("proj-a", "eu", "/path/to/key1.json");
  assert.notStrictEqual(client1, clientAfterClear, "Clearing cache must produce a fresh client instance");

  // 5. Model catalog returns models with prompt cache configured
  const models = getGoogleVertexModels(DEFAULT_GEMINI_CACHE_TTL_SECONDS);
  assert.ok(Array.isArray(models) && models.length > 0, "Models must be returned");

  console.log("gemini client memoization ok: instance reuse, profile isolation, header safety, and cache clearing verified");
}

function testMessageConversionAndThinkingConfig() {
  // 1. Tool result formatting: matching toolCallId and merging parallel tool calls
  const mockContext = {
    messages: [
      {
        role: "user",
        content: "Please check date and list files",
      },
      {
        role: "assistant",
        content: [
          {
            type: "thinking",
            thinking: "I should call date and ls",
            thinkingSignature: "sig_think_123",
          },
          {
            type: "toolCall",
            id: "call_date_1",
            name: "date",
            arguments: {},
          },
          {
            type: "toolCall",
            id: "call_ls_2",
            name: "ls",
            arguments: { path: "." },
          },
        ],
      },
      {
        role: "toolResult",
        toolCallId: "call_date_1",
        toolName: "date",
        content: [{ type: "text", text: "2026-11-04" }],
        isError: false,
      },
      {
        role: "toolResult",
        toolCallId: "call_ls_2",
        toolName: "ls",
        content: [{ type: "text", text: "file1.txt\nfile2.txt" }],
        isError: false,
      },
    ],
  };

  const converted = convertMessages(mockContext);
  assert.strictEqual(converted.length, 3, "Parallel tool calls must result in 3 turns: user, model, user");

  // Verify assistant turn preserves thinking with thought: true and thoughtSignature
  const assistantTurn = converted[1];
  assert.strictEqual(assistantTurn.role, "model");
  assert.strictEqual(assistantTurn.parts[0].thought, true);
  assert.strictEqual(assistantTurn.parts[0].thoughtSignature, "sig_think_123");
  assert.strictEqual(assistantTurn.parts[1].functionCall.id, "call_date_1");
  assert.strictEqual(assistantTurn.parts[2].functionCall.id, "call_ls_2");

  // Verify tool result turn merged both function responses into a SINGLE user turn
  const toolResponseTurn = converted[2];
  assert.strictEqual(toolResponseTurn.role, "user");
  assert.strictEqual(toolResponseTurn.parts.length, 2, "Consecutive tool responses must be merged into one turn");
  assert.strictEqual(toolResponseTurn.parts[0].functionResponse.id, "call_date_1");
  assert.strictEqual(toolResponseTurn.parts[0].functionResponse.response.output, "2026-11-04");
  assert.strictEqual(toolResponseTurn.parts[1].functionResponse.id, "call_ls_2");
  assert.strictEqual(toolResponseTurn.parts[1].functionResponse.response.output, "file1.txt\nfile2.txt");

  // 2. Multimodal input handling (user messages & tool results)
  const multimodalContext = {
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: "What is in this image?" },
          { type: "image", mimeType: "image/png", data: "base64data==" },
        ],
      },
    ],
  };
  const convertedMultimodal = convertMessages(multimodalContext);
  assert.strictEqual(convertedMultimodal.length, 1);
  assert.strictEqual(convertedMultimodal[0].parts.length, 2);
  assert.strictEqual(convertedMultimodal[0].parts[0].text, "What is in this image?");
  assert.deepStrictEqual(convertedMultimodal[0].parts[1].inlineData, {
    mimeType: "image/png",
    data: "base64data==",
  });

  const toolWithImageContext = {
    messages: [
      {
        role: "toolResult",
        toolCallId: "call_snap_1",
        toolName: "take_screenshot",
        content: [
          { type: "text", text: "Screenshot of dashboard" },
          { type: "image", mimeType: "image/png", data: "dashboard_base64==" },
        ],
        isError: false,
      },
    ],
  };
  const convertedToolWithImage = convertMessages(toolWithImageContext);
  assert.strictEqual(convertedToolWithImage[0].parts[0].functionResponse.response.output, "Screenshot of dashboard");
  assert.deepStrictEqual(convertedToolWithImage[0].parts[0].functionResponse.parts, [
    { inlineData: { mimeType: "image/png", data: "dashboard_base64==" } },
  ]);

  // 3. Thinking level mapping for Gemini 3+ vs 2.5
  assert.strictEqual(isGemini3OrNewer("gemini-3.8-flash"), true);
  assert.strictEqual(isGemini3OrNewer("gemini-3.5-flash-lite"), true);
  assert.strictEqual(isGemini3OrNewer("gemini-4-argon"), true);
  assert.strictEqual(isGemini3OrNewer("gemini-2.5-pro"), false);
  assert.strictEqual(isGemini3OrNewer("gemini-2.5-flash"), false);

  assert.strictEqual(mapThinkingLevel("low"), "LOW");
  assert.strictEqual(mapThinkingLevel("medium"), "MEDIUM");
  assert.strictEqual(mapThinkingLevel("high"), "HIGH");

  // 4. buildParams thinking config for Gemini 3+ vs 2.5
  const mockModelGemini3 = {
    id: "gemini-3.8-flash",
    provider: "google-vertex",
    reasoning: true,
  };
  const mockModelGemini25 = {
    id: "gemini-2.5-pro",
    provider: "google-vertex",
    reasoning: true,
  };

  const paramsWithLevel = buildParams(mockModelGemini3, { messages: [] }, {
    thinking: { enabled: true, level: "LOW" },
  });
  assert.deepStrictEqual(paramsWithLevel.config.thinkingConfig, {
    includeThoughts: true,
    thinkingLevel: "LOW",
  });

  // Gemini 3+ disabled thinking must NOT include thinkingBudget
  const paramsDisabledGemini3 = buildParams(mockModelGemini3, { messages: [] }, {
    thinking: { enabled: false },
  });
  assert.deepStrictEqual(paramsDisabledGemini3.config.thinkingConfig, {
    includeThoughts: false,
  });

  // Gemini 2.5 disabled thinking uses thinkingBudget: 0
  const paramsDisabledGemini25 = buildParams(mockModelGemini25, { messages: [] }, {
    thinking: { enabled: false },
  });
  assert.deepStrictEqual(paramsDisabledGemini25.config.thinkingConfig, {
    includeThoughts: false,
    thinkingBudget: 0,
  });

  // 5. Tool call and response IDs: omitted on Gemini 2.5, included on Gemini 3+
  const convertedGemini25 = convertMessages(mockContext, mockModelGemini25);
  assert.strictEqual(convertedGemini25[1].parts[1].functionCall.id, undefined, "Gemini 2.5 functionCall must not have id");
  assert.strictEqual(convertedGemini25[2].parts[0].functionResponse.id, undefined, "Gemini 2.5 functionResponse must not have id");

  const convertedGemini3 = convertMessages(mockContext, mockModelGemini3);
  assert.strictEqual(convertedGemini3[1].parts[1].functionCall.id, "call_date_1", "Gemini 3+ functionCall must have id");
  assert.strictEqual(convertedGemini3[2].parts[0].functionResponse.id, "call_date_1", "Gemini 3+ functionResponse must have id");

  console.log("gemini protocol & thinking fixes ok: tool IDs, turn merging, image support, and thinking configs verified");
}

await testClientMemoization();
testMessageConversionAndThinkingConfig();
