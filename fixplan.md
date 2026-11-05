# Gemini 4 Argon Latency & Compatibility Fix Plan

## Executive Summary
This document outlines the root causes, exact code modifications, configuration changes, and testing strategy required to eliminate the severe inference latency experienced with `gemini-4-argon` on Google Cloud Vertex AI within `custom-tweaked-extensions/NICK-pi-gcp-agent-platform`.

The investigation identified two primary classes of bottlenecks:
1. **Critical Code Flaws (`extensions/gcp-agent-platform/gemini.ts`)**:
   - **Unbounded Decodes**: Omission of `maxOutputTokens` allowed Argon's reasoning engine to generate up to 1,000,000 tokens in runaway loops.
   - **Forced `includeThoughts: true` & Transcript Bloat**: Misinterpreting Argon's minimum `"low"` level caused thinking to be marked enabled on every turn, streaming raw thought tokens over SSE and compounding tens of thousands of past thought tokens into future input prompts.
   - **Dropped `thoughtSignature`**: Empty text blocks carrying `thoughtSignature` were discarded on tool-calling turns, breaking multi-turn reasoning continuity.
   - **Legacy Tool Schemas**: Tool declarations used OpenAPI 3.0 `parameters` and ignored `const` values instead of modern `parametersJsonSchema` with `enum` rewriting.
2. **Configuration & Infrastructure Delays (`~/.pi/agent/gcp-agent-platform.json`)**:
   - **Disabled Cache Warming (`geminiCacheTtlSeconds: 0`)**: Disabled Pi's `CacheWarmer`, causing 100% cold prefill misses (15–30s delays) on any turn after 5 minutes of inactivity.
   - **EU Regional Data Residency Queuing (`location: "eu"`)**: Routing to `aiplatform.eu.rep.googleapis.com` where European accelerator capacity for frontier reasoning models faces substantial server-side scheduling queues.

---

## Detailed Action Plan

### Fix 1 (P0 — Code): Add `maxOutputTokens` Fallback to Prevent Unbounded Decodes
* **File**: `extensions/gcp-agent-platform/gemini.ts`
* **Problem**: Pi's agent loop passes `options` without `maxTokens`, expecting the provider to fall back to `model.maxTokens`. In `gemini.ts`, `streamSimpleGeminiVertex` passed `options` directly, and `buildParams` only set `generationConfig.maxOutputTokens` if `options.maxTokens` was defined. Because it was undefined, `maxOutputTokens` was omitted from the request payload. On Gemini 4 Argon (which supports 1M output tokens for long decodes), this allowed internal reasoning loops to run without an upper generation ceiling.
* **Modification**:
  In `streamSimpleGeminiVertex`, ensure `maxTokens` falls back to `options?.maxTokens ?? model.maxTokens`:
  ```ts
  const base: GoogleVertexOptions = {
    ...options,
    maxTokens: options?.maxTokens ?? model.maxTokens,
    toolChoice: options?.toolChoice,
  };
  ```

---

### Fix 2 (P0 — Code): Correct Reasoning Gate & Omit `includeThoughts: true` for Hidden/Disabled Thinking
* **File**: `extensions/gcp-agent-platform/gemini.ts`
* **Problem**: 
  1. Argon's catalog entry specifies `thinkingLevelMap: { off: null, minimal: null, low: "low", ... }`. Because `"off"` maps to `null`, `clampThinkingLevel(argon, "off")` returns `"low"`.
  2. In `streamSimpleGeminiVertex`, checking `if (clampedReasoning === "off")` failed when thinking was off/unspecified, falling through to `thinking: { enabled: true, level: "LOW" }`.
  3. In `buildParams`, `options.thinking?.enabled` caused `thinkingConfig = { includeThoughts: true }` to be sent unconditionally. Every raw thought token streamed over the wire, was recorded by Pi into session history, and was re-sent on subsequent turns as input tokens, exponentially bloating prompt prefill size.
* **Modification**:
  1. In `streamSimpleGeminiVertex`, explicitly gate thinking on user intent (`options?.reasoning`):
     ```ts
     if (!options?.reasoning) {
       return streamGeminiVertex(model, context, clientPromise, {
         ...base,
         thinking: { enabled: false },
       });
     }

     const clampedReasoning = clampThinkingLevel(model, options.reasoning);
     if (clampedReasoning === "off") {
       return streamGeminiVertex(model, context, clientPromise, {
         ...base,
         thinking: { enabled: false },
       });
     }

     if (isGemini3OrNewer(model.id)) {
       return streamGeminiVertex(model, context, clientPromise, {
         ...base,
         thinking: {
           enabled: true,
           level: mapThinkingLevel(clampedReasoning),
         },
       });
     }
     ```
  2. In `buildParams`, when `!options.thinking.enabled`, configure the model to run silently without streaming thoughts:
     ```ts
     } else if (model.reasoning && options.thinking && !options.thinking.enabled) {
       if (isGemini3OrNewer(model.id)) {
         const fallback = clampThinkingLevel(model, "off");
         if (fallback === "off") {
           config.thinkingConfig = { includeThoughts: false };
         } else {
           // For models like Argon where thinking cannot be completely off:
           // Set thinkingLevel: "LOW" internally, but OMIT includeThoughts: true so thoughts are not streamed.
           config.thinkingConfig = { thinkingLevel: "LOW" as any };
         }
       } else {
         config.thinkingConfig = { includeThoughts: false, thinkingBudget: 0 };
       }
     }
     ```

---

### Fix 3 (P1 — Code): Preserve Empty Text Parts Bearing `thoughtSignature`
* **File**: `extensions/gcp-agent-platform/gemini.ts`
* **Problem**: In `convertMessages`, `if (block.type === "text" && block.text)` dropped parts where `text` was an empty string (`""`). On tool-calling turns, Gemini 4 Argon frequently attaches its cryptographic `thoughtSignature` to an empty text block before the tool call. Dropping it breaks the reasoning continuity chain on subsequent turns, causing Vertex AI to halt, hang, or restart reasoning from scratch.
* **Modification**:
  In `convertMessages`, preserve empty text parts if a signature is present:
  ```ts
  if (block.type === "text") {
    const signature = block.textSignature;
    if ((!block.text || block.text.trim() === "") && !signature) continue;
    parts.push({
      text: block.text ?? "",
      ...(signature ? { thoughtSignature: signature } : {}),
    });
  }
  ```

---

### Fix 4 (P2 — Code): Modern `parametersJsonSchema` and `rewriteConstAsEnum` for Tools
* **File**: `extensions/gcp-agent-platform/gemini.ts`
* **Problem**: Tool definitions were passed using legacy OpenAPI 3.0 `parameters: tool.parameters`. On Vertex AI, Gemini ignores the modern JSON Schema `const` keyword (frequently used by tools) unless transformed into a single-element `enum: [value]`. Passing legacy `parameters` also forces a server-side translation preflight on Vertex AI.
* **Modification**:
  Add a self-contained helper in `gemini.ts` and target `parametersJsonSchema`:
  ```ts
  function rewriteConstAsEnum(schema: any): any {
    if (Array.isArray(schema)) return schema.map(rewriteConstAsEnum);
    if (typeof schema !== "object" || schema === null) return schema;

    const result: Record<string, any> = {};
    for (const [key, value] of Object.entries(schema)) {
      result[key] = rewriteConstAsEnum(value);
    }
    if ("const" in result && result.enum === undefined) {
      result.enum = [result.const];
      delete result.const;
    }
    return result;
  }

  function convertTools(tools: Tool[]): any[] {
    return [
      {
        functionDeclarations: tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          parametersJsonSchema: rewriteConstAsEnum(tool.parameters),
        })),
      },
    ];
  }
  ```

---

### Fix 5 (P1 — Configuration): Re-Enable Prompt Cache Warming
* **File**: `~/.pi/agent/gcp-agent-platform.json`
* **Problem**: `"geminiCacheTtlSeconds": 0` disables `promptCache` on the model object, which completely deactivates Pi's background `CacheWarmer`. Because Vertex AI only supports implicit caching for Gemini 4 Argon, any pause >5 minutes causes the server cache to expire, incurring full prefill latency (15–30s) on the next prompt.
* **Modification**:
  Set `geminiCacheTtlSeconds` to `300` (5 minutes) in `~/.pi/agent/gcp-agent-platform.json`:
  ```json
  "geminiCacheTtlSeconds": 300
  ```
  *(Or execute `/vertex-cache gemini 300` in Pi).*

---

### Fix 6 (P1 — Configuration): Switch Location from `eu` to `global`
* **File**: `~/.pi/agent/gcp-agent-platform.json`
* **Problem**: `"location": "eu"` routes requests to `https://aiplatform.eu.rep.googleapis.com/`. Frontier reasoning accelerator capacity in EU data residency clusters is heavily constrained compared to `global` and US regions, introducing multi-second server scheduling queue delays before generation begins.
* **Modification**:
  Set `location` to `"global"` in `~/.pi/agent/gcp-agent-platform.json`:
  ```json
  "location": "global"
  ```

---

## Test & Regression Strategy

### Updates to `scripts/gemini-cache.test.mjs`
Add automated unit tests verifying each patched behavior:
1. **`maxOutputTokens` propagation**: Verify `buildParams(mockModelGemini4, context, { maxTokens: 65536 })` sets `generationConfig.maxOutputTokens === 65536`.
2. **Disabled thinking on Argon**: Verify `buildParams(mockModelGemini4, context, { thinking: { enabled: false } })` emits `{ thinkingLevel: "LOW" }` and does **not** include `includeThoughts`.
3. **Empty text block signature preservation**: Verify `convertMessages` retains an assistant part with `{ type: "text", text: "", textSignature: "sig_abc" }` as `{ text: "", thoughtSignature: "sig_abc" }`.
4. **Tool schema `parametersJsonSchema` and `const` rewriting**: Verify `convertTools([{ parameters: { properties: { mode: { const: "fast" } } } }])` produces `parametersJsonSchema` with `enum: ["fast"]`.

### Execution
Run the full test suite in `custom-tweaked-extensions/NICK-pi-gcp-agent-platform`:
```bash
npm test
```

---

## Post-Compaction Implementation Checklist
- [ ] Edit `extensions/gcp-agent-platform/gemini.ts` to implement Fixes 1, 2, 3, and 4.
- [ ] Edit `scripts/gemini-cache.test.mjs` to add regression test assertions.
- [ ] Run `npm test` in `custom-tweaked-extensions/NICK-pi-gcp-agent-platform` to ensure all tests pass cleanly.
- [ ] Edit `~/.pi/agent/gcp-agent-platform.json` to set `"location": "global"` and `"geminiCacheTtlSeconds": 300`.
- [ ] Commit the changes to the git repository.
