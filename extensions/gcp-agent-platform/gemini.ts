/**
 * Gemini on Google Cloud Vertex AI / Gemini Enterprise Agent Platform.
 *
 * The parent extension owns Google Cloud project, region, strict auth credential
 * resolution, and the setup command. This module adapts Gemini Vertex AI models
 * with memoized client pooling so OAuth tokens are cached in memory across agent turns.
 */

import type {
  GenerateContentConfig,
  GenerateContentParameters,
  GoogleGenAI,
  HttpOptions,
  ThinkingConfig,
} from "@google/genai";
import {
  AssistantMessageEventStream,
  calculateCost,
  clampThinkingLevel,
  collapseSystemMessages,
  getCurrentTools,
  getInitialSystemMessage,
  getModels,
  getSystemMessageText,
  type Api,
  type AssistantMessage,
  type ImageContent,
  type Model,
  type ProviderHeaders,
  type SimpleStreamOptions,
  type StopReason,
  type StreamOptions,
  type TextContent,
  type ThinkingBudgets,
  type ThinkingContent,
  type Tool,
  type ToolCall,
  type TranscriptContext,
} from "@earendil-works/pi-ai/compat";
import {
  type ExtensionAPI,
  type ProviderModelConfig,
} from "@earendil-works/pi-coding-agent";

export const GEMINI_CACHE_MODELS = new Set([
  "gemini-3.5-flash-lite",
  "gemini-3.8-flash",
  "gemini-4-argon",
]);

export const DEFAULT_GEMINI_CACHE_TTL_SECONDS = 300;

const API_VERSION = "v1";
const VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform";

let toolCallCounter = 0;
let GoogleGenAIClass: typeof GoogleGenAI | undefined;

async function getGoogleGenAIClass(): Promise<typeof GoogleGenAI> {
  if (!GoogleGenAIClass) {
    const mod = await import("@google/genai");
    GoogleGenAIClass = mod.GoogleGenAI;
  }
  return GoogleGenAIClass;
}

type GeminiVertexModel = Omit<Model<Api>, "provider" | "baseUrl"> & {
  promptCache?: { short?: number; long?: number };
};

export function getGoogleVertexModels(
  geminiCacheTtlSeconds: number = DEFAULT_GEMINI_CACHE_TTL_SECONDS,
): GeminiVertexModel[] {
  const ttl =
    Number.isFinite(geminiCacheTtlSeconds) && geminiCacheTtlSeconds > 0
      ? Math.floor(geminiCacheTtlSeconds)
      : 0;
  return getModels("google-vertex").map(
    ({ provider: _provider, baseUrl: _baseUrl, ...model }) => ({
      ...model,
      promptCache:
        GEMINI_CACHE_MODELS.has(model.id) && ttl > 0
          ? { short: ttl, long: ttl }
          : undefined,
    }),
  );
}

const sharedClient = new Map<string, GoogleGenAI>();

export function clearGoogleVertexClientCache(): void {
  sharedClient.clear();
}

export async function getOrCreateGeminiVertexClient(
  project: string,
  location: string,
  keyFilename?: string,
  customHeaders?: ProviderHeaders,
): Promise<GoogleGenAI> {
  const GenAI = await getGoogleGenAIClass();

  if (customHeaders && Object.keys(customHeaders).length > 0) {
    return createGeminiVertexClient(GenAI, project, location, keyFilename, customHeaders);
  }

  const profileKey = `${project}:${location}:${keyFilename ?? "default"}`;
  let client = sharedClient.get(profileKey);
  if (!client) {
    client = createGeminiVertexClient(GenAI, project, location, keyFilename);
    sharedClient.set(profileKey, client);
  }
  return client;
}

function createGeminiVertexClient(
  GenAI: typeof GoogleGenAI,
  project: string,
  location: string,
  keyFilename?: string,
  headers?: ProviderHeaders,
): GoogleGenAI {
  const httpOptions: HttpOptions = {};
  if (headers && Object.keys(headers).length > 0) {
    httpOptions.headers = headers as Record<string, string>;
  }

  return new GenAI({
    vertexai: true,
    project,
    location,
    apiVersion: API_VERSION,
    ...(keyFilename
      ? {
          googleAuthOptions: {
            keyFilename,
            scopes: [VERTEX_SCOPE],
          },
        }
      : {}),
    ...(Object.keys(httpOptions).length > 0 ? { httpOptions } : {}),
  });
}

export interface GoogleVertexOptions extends StreamOptions {
  toolChoice?: "auto" | "none" | "any";
  thinking?: {
    enabled: boolean;
    budgetTokens?: number;
    level?: string;
  };
  project?: string;
  location?: string;
}

function isThinkingPart(part: any): boolean {
  return part.thought === true;
}

function retainThoughtSignature(current?: string, incoming?: string): string | undefined {
  return incoming ?? current;
}

function mapStopReason(reason?: string): StopReason {
  if (!reason) return "stop";
  switch (reason) {
    case "STOP":
      return "stop";
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "RECITATION":
      return "contentFilter";
    default:
      return "stop";
  }
}

function convertTools(tools: Tool[]): any[] {
  return [
    {
      functionDeclarations: tools.map((tool) => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      })),
    },
  ];
}

export function extractToolResultText(
  content: string | (TextContent | ImageContent)[],
): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((part): part is TextContent => (part as any).type === "text")
      .map((part) => (part as any).text)
      .join("\n");
  }
  return "";
}

export function convertMessages(context: TranscriptContext, model?: Model<Api>): any[] {
  const isGemini3 = model ? isGemini3OrNewer(model.id) : true;
  const contents: any[] = [];
  for (const msg of context.messages) {
    if (msg.role === "system") continue;
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        contents.push({ role: "user", parts: [{ text: msg.content }] });
      } else if (Array.isArray(msg.content)) {
        const parts: any[] = [];
        for (const item of msg.content) {
          if (item.type === "text" && item.text) {
            parts.push({ text: item.text });
          } else if (item.type === "image" && (item as any).data) {
            parts.push({
              inlineData: {
                mimeType: (item as any).mimeType,
                data: (item as any).data,
              },
            });
          }
        }
        if (parts.length > 0) contents.push({ role: "user", parts });
      }
    } else if (msg.role === "assistant") {
      const parts: any[] = [];
      for (const block of msg.content) {
        if (block.type === "text" && block.text) {
          parts.push({
            text: block.text,
            ...(block.textSignature ? { thoughtSignature: block.textSignature } : {}),
          });
        } else if (block.type === "thinking" && block.thinking) {
          parts.push({
            text: block.thinking,
            thought: true,
            ...(block.thinkingSignature ? { thoughtSignature: block.thinkingSignature } : {}),
          });
        } else if (block.type === "toolCall") {
          parts.push({
            functionCall: {
              ...(isGemini3 && block.id ? { id: block.id } : {}),
              name: block.name,
              args: block.arguments ?? {},
            },
            ...(block.thoughtSignature ? { thoughtSignature: block.thoughtSignature } : {}),
          });
        }
      }
      if (parts.length > 0) contents.push({ role: "model", parts });
    } else if (msg.role === "toolResult") {
      const resultText = extractToolResultText(msg.content);
      const imageParts = Array.isArray(msg.content)
        ? msg.content
            .filter((part): part is ImageContent => (part as any).type === "image" && Boolean((part as any).data))
            .map((part) => ({
              inlineData: {
                mimeType: (part as any).mimeType,
                data: (part as any).data,
              },
            }))
        : [];

      const functionResponsePart: any = {
        functionResponse: {
          name: msg.toolName ?? "",
          ...(isGemini3 && msg.toolCallId ? { id: msg.toolCallId } : {}),
          response: msg.isError ? { error: resultText } : { output: resultText },
          ...(imageParts.length > 0 ? { parts: imageParts } : {}),
        },
      };

      // Group consecutive tool responses into the same user turn
      const lastContent = contents[contents.length - 1];
      if (
        lastContent &&
        lastContent.role === "user" &&
        Array.isArray(lastContent.parts) &&
        lastContent.parts.some((p: any) => p.functionResponse)
      ) {
        lastContent.parts.push(functionResponsePart);
      } else {
        contents.push({ role: "user", parts: [functionResponsePart] });
      }
    }
  }
  return contents;
}

function getGoogleBudget(
  model: Model<Api>,
  level: string,
  customBudgets?: ThinkingBudgets,
): number {
  if (customBudgets && (customBudgets as any)[level] !== undefined) {
    return (customBudgets as any)[level]!;
  }
  if (model.id.includes("2.5-pro")) {
    const budgets: Record<string, number> = {
      minimal: 128,
      low: 2048,
      medium: 8192,
      high: 32768,
    };
    return budgets[level] ?? -1;
  }
  if (model.id.includes("2.5-flash")) {
    const budgets: Record<string, number> = {
      minimal: 128,
      low: 2048,
      medium: 8192,
      high: 24576,
    };
    return budgets[level] ?? -1;
  }
  return -1;
}

export function isGemini3OrNewer(modelId: string): boolean {
  const match = modelId.toLowerCase().match(/^gemini(?:-live)?-(\d+)/);
  if (!match) return false;
  return Number.parseInt(match[1], 10) >= 3;
}

export function mapThinkingLevel(level: string): string {
  switch (level.toLowerCase()) {
    case "minimal":
      return "MINIMAL";
    case "low":
      return "LOW";
    case "medium":
      return "MEDIUM";
    case "high":
    case "xhigh":
      return "HIGH";
    default:
      return "MEDIUM";
  }
}

export function buildParams(
  model: Model<Api>,
  context: TranscriptContext,
  options: GoogleVertexOptions = {},
): GenerateContentParameters {
  const contents = convertMessages(context, model);
  const initialSystemMessage = getInitialSystemMessage(context.messages);
  const currentTools = getCurrentTools(context.messages);

  const generationConfig: GenerateContentConfig = {};
  if (options.temperature !== undefined) {
    generationConfig.temperature = options.temperature;
  }
  if (options.maxTokens !== undefined) {
    generationConfig.maxOutputTokens = options.maxTokens;
  }

  const systemInstruction = initialSystemMessage
    ? getSystemMessageText(initialSystemMessage)
    : "";

  let functionCallingMode: "AUTO" | "ANY" | "NONE" | undefined;
  if (currentTools.length > 0) {
    if (options.toolChoice === "none") functionCallingMode = "NONE";
    else if (options.toolChoice === "any") functionCallingMode = "ANY";
    else functionCallingMode = "AUTO";
  }

  const config: GenerateContentConfig = {
    ...(Object.keys(generationConfig).length > 0 && generationConfig),
    ...(systemInstruction && { systemInstruction }),
    ...(currentTools.length > 0 && { tools: convertTools(currentTools) }),
    ...(functionCallingMode && {
      toolConfig: { functionCallingConfig: { mode: functionCallingMode as any } },
    }),
  };

  if (options.thinking?.enabled && model.reasoning) {
    const thinkingConfig: ThinkingConfig = { includeThoughts: true };
    if (isGemini3OrNewer(model.id)) {
      if (options.thinking.level !== undefined) {
        thinkingConfig.thinkingLevel = options.thinking.level as any;
      }
    } else if (options.thinking.budgetTokens !== undefined) {
      thinkingConfig.thinkingBudget = options.thinking.budgetTokens;
    }
    config.thinkingConfig = thinkingConfig;
  } else if (model.reasoning && options.thinking && !options.thinking.enabled) {
    if (isGemini3OrNewer(model.id)) {
      config.thinkingConfig = { includeThoughts: false };
    } else {
      config.thinkingConfig = { includeThoughts: false, thinkingBudget: 0 };
    }
  }

  if (options.signal) {
    if (options.signal.aborted) {
      throw new Error("Request aborted");
    }
    config.abortSignal = options.signal;
  }

  return {
    model: model.id,
    contents,
    config,
  };
}

function streamGeminiVertex(
  model: Model<Api>,
  context: TranscriptContext,
  clientPromise: Promise<GoogleGenAI>,
  options?: GoogleVertexOptions,
): AssistantMessageEventStream {
  const stream = new AssistantMessageEventStream();
  const normalizedContext = collapseSystemMessages(context);

  (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: "google-vertex" as Api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "pending",
      timestamp: Date.now(),
    };

    try {
      const client = await clientPromise;
      let params = buildParams(model, normalizedContext, options);
      const nextParams = await options?.onPayload?.(params, model);
      if (nextParams !== undefined) {
        params = nextParams as GenerateContentParameters;
      }

      const googleStream = await client.models.generateContentStream(params);

      stream.push({ type: "start", partial: output });
      let currentBlock: TextContent | ThinkingContent | null = null;
      const blocks = output.content;
      const blockIndex = () => blocks.length - 1;

      for await (const chunk of googleStream) {
        await options?.onProviderStreamEvent?.(chunk, model);
        output.responseId ||= chunk.responseId;
        const candidate = chunk.candidates?.[0];
        if (candidate?.content?.parts) {
          for (const part of candidate.content.parts) {
            if (part.text !== undefined) {
              const isThinking = isThinkingPart(part);
              if (
                !currentBlock ||
                (isThinking && currentBlock.type !== "thinking") ||
                (!isThinking && currentBlock.type !== "text")
              ) {
                if (currentBlock) {
                  if (currentBlock.type === "text") {
                    stream.push({
                      type: "text_end",
                      contentIndex: blocks.length - 1,
                      content: currentBlock.text,
                      partial: output,
                    });
                  } else {
                    stream.push({
                      type: "thinking_end",
                      contentIndex: blockIndex(),
                      content: currentBlock.thinking,
                      partial: output,
                    });
                  }
                }
                if (isThinking) {
                  currentBlock = {
                    type: "thinking",
                    thinking: "",
                    thinkingSignature: undefined,
                  };
                  output.content.push(currentBlock);
                  stream.push({
                    type: "thinking_start",
                    contentIndex: blockIndex(),
                    partial: output,
                  });
                } else {
                  currentBlock = { type: "text", text: "" };
                  output.content.push(currentBlock);
                  stream.push({
                    type: "text_start",
                    contentIndex: blockIndex(),
                    partial: output,
                  });
                }
              }
              if (currentBlock.type === "thinking") {
                currentBlock.thinking += part.text;
                currentBlock.thinkingSignature = retainThoughtSignature(
                  currentBlock.thinkingSignature,
                  part.thoughtSignature,
                );
                stream.push({
                  type: "thinking_delta",
                  contentIndex: blockIndex(),
                  delta: part.text,
                  partial: output,
                });
              } else {
                currentBlock.text += part.text;
                currentBlock.textSignature = retainThoughtSignature(
                  currentBlock.textSignature,
                  part.thoughtSignature,
                );
                stream.push({
                  type: "text_delta",
                  contentIndex: blockIndex(),
                  delta: part.text,
                  partial: output,
                });
              }
            }

            if (part.functionCall) {
              if (currentBlock) {
                if (currentBlock.type === "text") {
                  stream.push({
                    type: "text_end",
                    contentIndex: blockIndex(),
                    content: currentBlock.text,
                    partial: output,
                  });
                } else {
                  stream.push({
                    type: "thinking_end",
                    contentIndex: blockIndex(),
                    content: currentBlock.thinking,
                    partial: output,
                  });
                }
                currentBlock = null;
              }

              const providedId = part.functionCall.id;
              const needsNewId =
                !providedId ||
                output.content.some(
                  (b) => b.type === "toolCall" && b.id === providedId,
                );
              const toolCallId = needsNewId
                ? `${part.functionCall.name}_${Date.now()}_${++toolCallCounter}`
                : providedId;

              const toolCall: ToolCall = {
                type: "toolCall",
                id: toolCallId,
                name: part.functionCall.name || "",
                arguments: (part.functionCall.args as Record<string, any>) ?? {},
                ...(part.thoughtSignature && {
                  thoughtSignature: part.thoughtSignature,
                }),
              };

              output.content.push(toolCall);
              stream.push({
                type: "toolcall_start",
                contentIndex: blockIndex(),
                partial: output,
              });
              stream.push({
                type: "toolcall_delta",
                contentIndex: blockIndex(),
                delta: JSON.stringify(toolCall.arguments),
                partial: output,
              });
              stream.push({
                type: "toolcall_end",
                contentIndex: blockIndex(),
                toolCall,
                partial: output,
              });
            }
          }
        }

        if (candidate?.finishReason) {
          output.rawStopReason = candidate.finishReason;
          output.stopReason = mapStopReason(candidate.finishReason);
          if (
            output.content.some((b) => b.type === "toolCall") &&
            output.stopReason === "stop"
          ) {
            output.stopReason = "toolUse";
          }
        }

        if (chunk.usageMetadata) {
          output.usage = {
            input:
              (chunk.usageMetadata.promptTokenCount || 0) -
              (chunk.usageMetadata.cachedContentTokenCount || 0),
            output:
              (chunk.usageMetadata.candidatesTokenCount || 0) +
              (chunk.usageMetadata.thoughtsTokenCount || 0),
            cacheRead: chunk.usageMetadata.cachedContentTokenCount || 0,
            cacheWrite: 0,
            reasoning: chunk.usageMetadata.thoughtsTokenCount || 0,
            totalTokens: chunk.usageMetadata.totalTokenCount || 0,
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              total: 0,
            },
          };
          calculateCost(model, output.usage);
        }
      }

      if (currentBlock) {
        if (currentBlock.type === "text") {
          stream.push({
            type: "text_end",
            contentIndex: blockIndex(),
            content: currentBlock.text,
            partial: output,
          });
        } else {
          stream.push({
            type: "thinking_end",
            contentIndex: blockIndex(),
            content: currentBlock.thinking,
            partial: output,
          });
        }
      }

      if (options?.signal?.aborted) {
        throw new Error("Request was aborted");
      }

      if (output.stopReason === "pending") {
        throw new Error("Google Vertex stream ended without a finish reason");
      }
      if (output.stopReason === "aborted" || output.stopReason === "error") {
        const errorMessage = output.rawStopReason
          ? `Provider stopped with: ${output.rawStopReason}`
          : "An unknown error occurred";
        throw new Error(errorMessage);
      }

      stream.push({ type: "done", reason: output.stopReason, message: output });
      stream.end();
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();

  return stream;
}

export function streamSimpleGeminiVertex(
  model: Model<Api>,
  context: TranscriptContext,
  clientPromise: Promise<GoogleGenAI>,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const base: GoogleVertexOptions = {
    ...options,
    toolChoice: options?.toolChoice,
  };

  const requestedReasoning = options?.reasoning ?? (model.reasoning ? "off" : undefined);
  if (!requestedReasoning) {
    return streamGeminiVertex(model, context, clientPromise, {
      ...base,
      thinking: { enabled: false },
    });
  }

  const clampedReasoning = clampThinkingLevel(model, requestedReasoning);
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

  const budget = getGoogleBudget(model, clampedReasoning, options.thinkingBudgets);

  return streamGeminiVertex(model, context, clientPromise, {
    ...base,
    thinking: {
      enabled: true,
      budgetTokens: budget > 0 ? budget : undefined,
    },
  });
}

export function registerGoogleVertexProvider(
  pi: ExtensionAPI,
  project: string | undefined,
  location: string,
  _baseUrl: string,
  geminiCacheTtlSeconds: number = DEFAULT_GEMINI_CACHE_TTL_SECONDS,
  resolveCredentialsFile?: () => string,
): void {
  sharedClient.clear();

  if (!project) {
    console.warn(
      "[pi-gcp-agent-platform] Gemini Vertex disabled: configure Google Cloud project first",
    );
    return;
  }

  const models = getGoogleVertexModels(
    geminiCacheTtlSeconds,
  ) as unknown as ProviderModelConfig[];
  if (models.length === 0) return;

  const sdkTemplateBaseUrl = "https://{location}-aiplatform.googleapis.com";

  pi.registerProvider("google-vertex", {
    baseUrl: sdkTemplateBaseUrl,
    apiKey: "$GOOGLE_CLOUD_PROJECT",
    api: "google-vertex",
    models,
    streamSimple: (
      model: Model<Api>,
      context: TranscriptContext,
      options?: SimpleStreamOptions,
    ) => {
      const keyFilename = resolveCredentialsFile
        ? resolveCredentialsFile()
        : process.env.GOOGLE_APPLICATION_CREDENTIALS;

      const clientPromise = getOrCreateGeminiVertexClient(
        project,
        location,
        keyFilename,
        options?.headers,
      );

      return streamSimpleGeminiVertex(model, context, clientPromise, options);
    },
  });
}
