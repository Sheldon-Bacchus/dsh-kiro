/**
 * Kiro `/v1/responses` (CreateResponse) transport channel, feature-flag probe,
 * request translation, error normalization, and SSE-to-StreamChunk parser.
 *
 * Accounts assigned to Kiro's `create_response_api` cohort receive fixed
 * `USER_REQUEST_RATE_EXCEEDED` (429) responses on the legacy
 * `generateAssistantResponse` endpoint regardless of quota. This module probes
 * `featureConfiguration` and speaks the streaming `/v1/responses` protocol
 * while letting {@link KiroAdapter} fall back cleanly on failure.
 *
 * @module dsh-kiro/create-response
 */
import { LlmError } from '@deepseek-ai/dsh-llm';
import type { StreamChunk } from '@deepseek-ai/dsh-llm';
import type { KiroToken } from './auth.ts';
import * as transport from './transport.ts';
import type { TransportRequest, TransportResponse } from './transport.ts';
import type { WireRequest } from './types.ts';
/** Adapter mode controlling whether the CreateResponse channel is used. */
export type CreateResponseApiMode = 'auto' | 'on' | 'off';
/** Positive/settled `featureConfiguration` probe cache TTL (15 minutes). */
export declare const FEATURE_CONFIG_TTL_MS: number;
/** Failed `featureConfiguration` probe cache TTL (60 seconds). */
export declare const FEATURE_CONFIG_FAILURE_TTL_MS: number;
/** HTTP timeout for `featureConfiguration` probes (5 seconds). */
export declare const FEATURE_CONFIG_TIMEOUT_MS = 5000;
/** One `input_text` or `input_image` content part on a Responses user message. */
export type CreateResponseUserContentPart = {
    type: 'input_text';
    text: string;
} | {
    type: 'input_image';
    image_url: string;
};
/** One `output_text` content part on a replayed Responses assistant message. */
export interface CreateResponseAssistantContentPart {
    type: 'output_text';
    text: string;
}
/** One entry in the Responses API `input` array. */
export type CreateResponseInputItem = {
    type: 'message';
    role: 'user';
    content: CreateResponseUserContentPart[];
} | {
    type: 'message';
    role: 'assistant';
    content: CreateResponseAssistantContentPart[];
} | {
    type: 'function_call';
    call_id: string;
    name: string;
    arguments: string;
} | {
    type: 'function_call_output';
    call_id: string;
    output: string;
};
/** One function tool schema in the Responses API shape. */
export interface CreateResponseTool {
    type: 'function';
    name: string;
    description: string;
    parameters: Record<string, unknown>;
}
/** Request payload accepted by `POST https://runtime.<region>.kiro.dev/v1/responses`. */
export interface CreateResponseRequest {
    model: string;
    stream: true;
    profileArn?: string;
    input: CreateResponseInputItem[];
    tools?: CreateResponseTool[];
}
/** One decoded SSE event frame. */
export interface SseEvent {
    event?: string;
    data: string;
}
/** Injectable GET JSON transport matching `transport.getJson`. */
export type FeatureConfigGetJson = typeof transport.getJson;
/** Injectable POST transport matching `transport.post`. */
export type CreateResponsePost = (options: TransportRequest) => Promise<TransportResponse>;
/** Parameters for {@link probeFeatureConfiguration}. */
export interface FeatureConfigProbeParams {
    token: KiroToken;
    profileArn?: string;
    region: string;
    proxyUrl?: string;
    signal?: AbortSignal;
    requestJson?: FeatureConfigGetJson;
    now?: () => number;
    timeoutMs?: number;
}
/** Parameters for {@link streamCreateResponse}. */
export interface StreamCreateResponseParams {
    token: KiroToken;
    region: string;
    profileArn?: string;
    wireRequest: WireRequest;
    proxyUrl?: string;
    signal: AbortSignal;
    onActivity?: () => void;
    postRequest?: CreateResponsePost;
}
/** Build the CreateResponse endpoint URL for a region. */
export declare function createResponseEndpoint(region: string): string;
/** Build the featureConfiguration endpoint URL for a region and IDE client version. */
export declare function featureConfigEndpoint(region: string, version?: string): string;
/**
 * Resolve the `TokenType` header required by `/v1/responses` and `featureConfiguration`.
 * Builder ID and IdC credentials use `SSO_OIDC`; other auth methods follow the
 * existing adapter rules (`API_KEY`, `EXTERNAL_IDP`).
 */
export declare function createResponseTokenTypeHeaders(token: Pick<KiroToken, 'authMethod'>): Record<string, string>;
/** Build the request headers required by `POST /v1/responses`. */
export declare function createResponseHeaders(token: Pick<KiroToken, 'accessToken' | 'authMethod'>, profileArn?: string): Record<string, string>;
/** Build the cache key for `featureConfiguration` probes (`accountId + profileArn + region`). */
export declare function featureConfigCacheKey(token: Pick<KiroToken, 'accessToken' | 'accountId'>, profileArn: string | undefined, region: string): string;
/** Clear all cached `featureConfiguration` probe results (intended for tests). */
export declare function clearFeatureConfigCache(): void;
/**
 * Evaluate a `featureConfiguration` JSON response body.
 *
 * Kiro returns `{ configuration: { "<sha256key>": boolean, ... } }` where
 * feature flag names are hashed on the wire without a guaranteed public salt or
 * preimage. Rather than hardcoding a hash key that could drift, we treat any
 * `true` flag in `configuration` as a signal to attempt `/v1/responses`,
 * relying on {@link KiroAdapter}'s automatic fallback to the legacy endpoint if
 * `/v1/responses` rejects or fails.
 */
export declare function isCreateResponseFeatureEnabled(body: unknown): boolean;
/**
 * Probe `GET /featureConfiguration` with caching (15 min on 200, 60s on failure, 5s timeout).
 * Returns `true` when `/v1/responses` should be attempted, `false` otherwise.
 */
export declare function probeFeatureConfiguration(params: FeatureConfigProbeParams): Promise<boolean>;
/**
 * Convert a serialized legacy `WireRequest` (from `src/serialize.ts`) into a
 * `/v1/responses` request payload.
 */
export declare function convertWireRequestToCreateResponse(wire: WireRequest): CreateResponseRequest;
/**
 * Normalize a non-2xx `/v1/responses` HTTP response into an {@link LlmError}.
 */
export declare function parseCreateResponseError(status: number, bodyText: string, headers?: Record<string, string | string[] | undefined>): LlmError;
/**
 * Incrementally decode a `text/event-stream` byte stream into {@link SseEvent}s.
 */
export declare function parseSseEvents(body: AsyncIterable<Uint8Array>, onActivity?: () => void): AsyncGenerator<SseEvent>;
/**
 * Translate a `/v1/responses` SSE byte stream into DSH {@link StreamChunk}s.
 *
 * Emits `block-start`, delta chunks (`text-delta`, `reasoning-delta`,
 * `tool-call-delta`), `block-end`, `usage`, and terminal `finish`. Throws
 * {@link LlmError} on SSE error events, malformed JSON, or truncated streams
 * that end before `response.completed` / `response.incomplete`.
 */
export declare function translateCreateResponseSse(body: AsyncIterable<Uint8Array>, onActivity?: () => void): AsyncGenerator<StreamChunk>;
/**
 * Send one streaming request to `POST /v1/responses` and translate its SSE stream.
 */
export declare function streamCreateResponse(params: StreamCreateResponseParams): AsyncGenerator<StreamChunk>;
