import assert from "node:assert/strict";
import {
  clearGoogleVertexClientCache,
  getOrCreateGeminiVertexClient,
  getGoogleVertexModels,
  DEFAULT_GEMINI_CACHE_TTL_SECONDS,
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

await testClientMemoization();
