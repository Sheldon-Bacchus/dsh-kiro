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

import { createHash } from 'node:crypto'
import {
  CONTEXT_WINDOW_EXCEEDED_CODE,
  EMPTY_RESPONSE_CODE,
  isContextWindowExceededError,
  isQuotaExceededError,
  LlmError,
  ProviderRequestId,
  QUOTA_EXCEEDED_CODE,
  ToolCallId,
} from '@deepseek-ai/dsh-llm'
import type { ContentBlock, FinishReason, StreamChunk, TokenUsage } from '@deepseek-ai/dsh-llm'
import type { KiroToken } from './auth.ts'
import { kiroClientVersion, kiroUserAgent } from './client-id.ts'
import * as transport from './transport.ts'
import type { TransportRequest, TransportResponse } from './transport.ts'
import type { WireRequest, WireUserInputMessage } from './types.ts'

/** Adapter mode controlling whether the CreateResponse channel is used. */
export type CreateResponseApiMode = 'auto' | 'on' | 'off'

/** Positive/settled `featureConfiguration` probe cache TTL (15 minutes). */
export const FEATURE_CONFIG_TTL_MS = 15 * 60 * 1000
/** Failed `featureConfiguration` probe cache TTL (60 seconds). */
export const FEATURE_CONFIG_FAILURE_TTL_MS = 60 * 1000
/** HTTP timeout for `featureConfiguration` probes (5 seconds). */
export const FEATURE_CONFIG_TIMEOUT_MS = 5_000

const ORIGIN = 'AI_EDITOR'
const AGENT_MODE = 'vibe'
const TOOL_RESULTS_ONLY_PLACEHOLDER = 'Tool results provided.'
const ACKNOWLEDGE_PADDING_TEXT = 'understood'

const CONTEXT_OVERFLOW_REASONS = ['CONTENT_LENGTH_EXCEEDS_THRESHOLD', 'CONVERSATION_LIMIT_EXCEEDED'] as const
const QUOTA_REASONS = ['MONTHLY_REQUEST_COUNT', 'DAILY_REQUEST_COUNT', 'OVERAGE_REQUEST_LIMIT_EXCEEDED'] as const

/** One `input_text` or `input_image` content part on a Responses user message. */
export type CreateResponseUserContentPart =
  | { type: 'input_text'; text: string }
  | { type: 'input_image'; image_url: string }

/** One `output_text` content part on a replayed Responses assistant message. */
export interface CreateResponseAssistantContentPart {
  type: 'output_text'
  text: string
}

/** One entry in the Responses API `input` array. */
export type CreateResponseInputItem =
  | { type: 'message'; role: 'user'; content: CreateResponseUserContentPart[] }
  | { type: 'message'; role: 'assistant'; content: CreateResponseAssistantContentPart[] }
  | { type: 'function_call'; call_id: string; name: string; arguments: string }
  | { type: 'function_call_output'; call_id: string; output: string }

/** One function tool schema in the Responses API shape. */
export interface CreateResponseTool {
  type: 'function'
  name: string
  description: string
  parameters: Record<string, unknown>
}

/** Request payload accepted by `POST https://runtime.<region>.kiro.dev/v1/responses`. */
export interface CreateResponseRequest {
  model: string
  stream: true
  profileArn?: string
  input: CreateResponseInputItem[]
  tools?: CreateResponseTool[]
}

/** One decoded SSE event frame. */
export interface SseEvent {
  event?: string
  data: string
}

/** Injectable GET JSON transport matching `transport.getJson`. */
export type FeatureConfigGetJson = typeof transport.getJson
/** Injectable POST transport matching `transport.post`. */
export type CreateResponsePost = (options: TransportRequest) => Promise<TransportResponse>

/** Parameters for {@link probeFeatureConfiguration}. */
export interface FeatureConfigProbeParams {
  token: KiroToken
  profileArn?: string
  region: string
  proxyUrl?: string
  signal?: AbortSignal
  requestJson?: FeatureConfigGetJson
  now?: () => number
  timeoutMs?: number
}

/** Parameters for {@link streamCreateResponse}. */
export interface StreamCreateResponseParams {
  token: KiroToken
  region: string
  profileArn?: string
  wireRequest: WireRequest
  proxyUrl?: string
  signal: AbortSignal
  onActivity?: () => void
  postRequest?: CreateResponsePost
}

interface ProbeCacheEntry {
  enabled: boolean
  expiresAt: number
}

const featureConfigCache = new Map<string, ProbeCacheEntry>()

function defaultGetJson(): FeatureConfigGetJson | undefined {
  try {
    return typeof transport.getJson === 'function' ? transport.getJson : undefined
  } catch {
    return undefined
  }
}

function defaultPost(): CreateResponsePost {
  return transport.post
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function nonNegativeInt(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

/** Build the CreateResponse endpoint URL for a region. */
export function createResponseEndpoint(region: string): string {
  return `https://runtime.${region}.kiro.dev/v1/responses`
}

/** Build the featureConfiguration endpoint URL for a region and IDE client version. */
export function featureConfigEndpoint(region: string, version: string = kiroClientVersion()): string {
  const params = new URLSearchParams({ origin: ORIGIN, version })
  return `https://runtime.${region}.kiro.dev/featureConfiguration?${params.toString()}`
}

/**
 * Resolve the `TokenType` header required by `/v1/responses` and `featureConfiguration`.
 * Builder ID and IdC credentials use `SSO_OIDC`; other auth methods follow the
 * existing adapter rules (`API_KEY`, `EXTERNAL_IDP`).
 */
export function createResponseTokenTypeHeaders(token: Pick<KiroToken, 'authMethod'>): Record<string, string> {
  if (token.authMethod === 'builder-id' || token.authMethod === 'idc') {
    return { TokenType: 'SSO_OIDC' }
  }
  if (token.authMethod === 'api_key') return { TokenType: 'API_KEY' }
  if (token.authMethod === 'external_idp') return { TokenType: 'EXTERNAL_IDP' }
  return {}
}

/** Build the request headers required by `POST /v1/responses`. */
export function createResponseHeaders(
  token: Pick<KiroToken, 'accessToken' | 'authMethod'>,
  profileArn?: string,
): Record<string, string> {
  const ua = kiroUserAgent()
  return {
    authorization: `Bearer ${token.accessToken}`,
    'content-type': 'application/json',
    accept: 'text/event-stream',
    'user-agent': ua,
    'x-amz-user-agent': ua,
    'x-amzn-kiro-origin': ORIGIN,
    'x-amzn-kiro-agent-mode': AGENT_MODE,
    ...profileArn === undefined ? {} : {
      'x-amzn-kiro-profile': profileArn,
      'x-amzn-kiro-profile-arn': profileArn,
    },
    ...createResponseTokenTypeHeaders(token),
  }
}

/** Build the cache key for `featureConfiguration` probes (`accountId + profileArn + region`). */
export function featureConfigCacheKey(
  token: Pick<KiroToken, 'accessToken' | 'accountId'>,
  profileArn: string | undefined,
  region: string,
): string {
  const account = token.accountId
    ?? `token:${createHash('sha256').update(token.accessToken).digest('hex').slice(0, 16)}`
  return `${account}\u0000${profileArn ?? ''}\u0000${region}`
}

/** Clear all cached `featureConfiguration` probe results (intended for tests). */
export function clearFeatureConfigCache(): void {
  featureConfigCache.clear()
}

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
export function isCreateResponseFeatureEnabled(body: unknown): boolean {
  const root = asRecord(body)
  const config = asRecord(root?.configuration)
  if (config === undefined) return false
  return Object.values(config).some(value => value === true)
}

/**
 * Probe `GET /featureConfiguration` with caching (15 min on 200, 60s on failure, 5s timeout).
 * Returns `true` when `/v1/responses` should be attempted, `false` otherwise.
 */
export async function probeFeatureConfiguration(params: FeatureConfigProbeParams): Promise<boolean> {
  const now = params.now ?? Date.now
  const currentTime = now()
  const key = featureConfigCacheKey(params.token, params.profileArn, params.region)
  const cached = featureConfigCache.get(key)
  if (cached !== undefined && cached.expiresAt > currentTime) {
    return cached.enabled
  }

  const requestJson = params.requestJson ?? defaultGetJson()
  if (requestJson === undefined) {
    return false
  }

  const timeoutMs = params.timeoutMs ?? FEATURE_CONFIG_TIMEOUT_MS
  const timeoutSignal = AbortSignal.timeout(timeoutMs)
  const probeSignal = params.signal === undefined
    ? timeoutSignal
    : AbortSignal.any([params.signal, timeoutSignal])

  const ua = kiroUserAgent()
  const headers: Record<string, string> = {
    authorization: `Bearer ${params.token.accessToken}`,
    'user-agent': ua,
    'x-amz-user-agent': ua,
    'x-amzn-kiro-origin': ORIGIN,
    'x-amzn-kiro-agent-mode': AGENT_MODE,
    ...params.profileArn === undefined ? {} : { 'x-amzn-kiro-profile-arn': params.profileArn },
    ...createResponseTokenTypeHeaders(params.token),
  }

  try {
    const { status, body } = await requestJson(
      featureConfigEndpoint(params.region),
      headers,
      params.proxyUrl,
      probeSignal,
    )
    const root = asRecord(body)
    const config = asRecord(root?.configuration)
    if (status === 200 && config !== undefined) {
      const enabled = isCreateResponseFeatureEnabled(body)
      featureConfigCache.set(key, { enabled, expiresAt: now() + FEATURE_CONFIG_TTL_MS })
      return enabled
    }
    featureConfigCache.set(key, { enabled: false, expiresAt: now() + FEATURE_CONFIG_FAILURE_TTL_MS })
    return false
  } catch (error: unknown) {
    if (params.signal?.aborted) {
      throw new LlmError('Kiro request aborted by caller', 'ABORTED', { cause: error })
    }
    featureConfigCache.set(key, { enabled: false, expiresAt: now() + FEATURE_CONFIG_FAILURE_TTL_MS })
    return false
  }
}

function appendUserMessageItems(
  message: WireUserInputMessage,
  input: CreateResponseInputItem[],
): void {
  const results = message.userInputMessageContext?.toolResults ?? []
  for (const result of results) {
    const output = result.content.map(part => part.text).join('\n')
    input.push({
      type: 'function_call_output',
      call_id: result.toolUseId,
      output,
    })
  }

  const images = message.images ?? []
  const isSyntheticToolPlaceholder = results.length > 0
    && message.content === TOOL_RESULTS_ONLY_PLACEHOLDER
    && images.length === 0
  if (isSyntheticToolPlaceholder) return

  const content: CreateResponseUserContentPart[] = []
  if (message.content.length > 0) {
    content.push({ type: 'input_text', text: message.content })
  }
  for (const image of images) {
    content.push({
      type: 'input_image',
      image_url: `data:image/${image.format};base64,${image.source.bytes}`,
    })
  }
  if (content.length > 0) {
    input.push({ type: 'message', role: 'user', content })
  }
}

/**
 * Convert a serialized legacy `WireRequest` (from `src/serialize.ts`) into a
 * `/v1/responses` request payload.
 */
export function convertWireRequestToCreateResponse(wire: WireRequest): CreateResponseRequest {
  const input: CreateResponseInputItem[] = []
  for (const entry of wire.conversationState.history ?? []) {
    if ('userInputMessage' in entry) {
      appendUserMessageItems(entry.userInputMessage, input)
      continue
    }
    const assistant = entry.assistantResponseMessage
    const toolUses = assistant.toolUses ?? []
    const isSyntheticPadding = toolUses.length > 0 && assistant.content === ACKNOWLEDGE_PADDING_TEXT
    if (assistant.content.length > 0 && !isSyntheticPadding) {
      input.push({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: assistant.content }],
      })
    }
    for (const use of toolUses) {
      input.push({
        type: 'function_call',
        call_id: use.toolUseId,
        name: use.name,
        arguments: typeof use.input === 'string' ? use.input : JSON.stringify(use.input ?? {}),
      })
    }
  }

  const currentUser = wire.conversationState.currentMessage.userInputMessage
  appendUserMessageItems(currentUser, input)

  const tools: CreateResponseTool[] = (currentUser.userInputMessageContext?.tools ?? []).map(tool => ({
    type: 'function',
    name: tool.toolSpecification.name,
    description: tool.toolSpecification.description,
    parameters: tool.toolSpecification.inputSchema.json,
  }))

  return {
    model: currentUser.modelId,
    stream: true,
    ...wire.profileArn === undefined ? {} : { profileArn: wire.profileArn },
    input,
    ...tools.length === 0 ? {} : { tools },
  }
}

function mapCreateResponseErrorCode(
  status: number,
  errorType: string | undefined,
  errorCode: string | undefined,
  message: string,
  rawBody: string,
): string {
  const combined = `${message}\n${rawBody}`
  if (
    errorCode === 'context_length_exceeded'
    || CONTEXT_OVERFLOW_REASONS.some(reason => combined.includes(reason))
    || isContextWindowExceededError(combined)
  ) {
    return CONTEXT_WINDOW_EXCEEDED_CODE
  }
  if (
    errorCode === 'insufficient_quota'
    || errorType === 'insufficient_quota'
    || (!combined.includes('CONVERSATION_LIMIT_EXCEEDED')
      && (QUOTA_REASONS.some(reason => combined.includes(reason)) || isQuotaExceededError(combined)))
  ) {
    return QUOTA_EXCEEDED_CODE
  }

  const normalizedTag = (errorCode ?? errorType ?? '').trim().toLowerCase()
  switch (normalizedTag) {
    case 'rate_limit_exceeded':
    case 'too_many_requests':
      return 'RATE_LIMIT'
    case 'server_is_overloaded':
    case 'overloaded_error':
    case 'server_error':
    case 'internal_error':
      return 'SERVER'
    case 'authentication_error':
    case 'invalid_api_key':
    case 'unauthorized':
      return 'AUTH'
    case 'permission_error':
    case 'forbidden':
      return 'FORBIDDEN'
    case 'model_not_found':
    case 'invalid_model':
    case 'invalid_model_id':
      return 'INVALID_MODEL'
    case 'invalid_request':
    case 'invalid_request_error':
      return 'INVALID_REQUEST'
    default:
      break
  }

  if (status === 401) return 'AUTH'
  if (status === 403) return combined.includes('bearer token') ? 'AUTH' : 'FORBIDDEN'
  if (status === 402) return QUOTA_EXCEEDED_CODE
  if (status === 429) return 'RATE_LIMIT'
  if (status === 400) return combined.includes('INVALID_MODEL_ID') ? 'INVALID_MODEL' : 'INVALID_REQUEST'
  if (status >= 500) return 'SERVER'
  return `HTTP_${status}`
}

/**
 * Normalize a non-2xx `/v1/responses` HTTP response into an {@link LlmError}.
 */
export function parseCreateResponseError(
  status: number,
  bodyText: string,
  headers?: Record<string, string | string[] | undefined>,
): LlmError {
  let message = `Kiro CreateResponse API error (HTTP ${status})`
  let errorType: string | undefined
  let errorCode: string | undefined
  let retryAfterMs: number | undefined

  try {
    const parsed = asRecord(JSON.parse(bodyText))
    const err = asRecord(parsed?.error) ?? parsed
    if (typeof err?.message === 'string' && err.message.length > 0) message = err.message
    if (typeof err?.type === 'string' && err.type.length > 0) errorType = err.type
    if (typeof err?.code === 'string' && err.code.length > 0) errorCode = err.code
    const rawRetry = err?.retryAfterMilliseconds ?? parsed?.retryAfterMilliseconds
    if (typeof rawRetry === 'number' && Number.isFinite(rawRetry) && rawRetry > 0) {
      retryAfterMs = rawRetry
    }
  } catch {
    // Non-JSON error bodies still map via HTTP status below.
  }

  const code = mapCreateResponseErrorCode(status, errorType, errorCode, message, bodyText)
  const rawRequestId = headers?.['x-amzn-requestid'] ?? headers?.['x-request-id']
  const requestId = typeof rawRequestId === 'string' && rawRequestId.length > 0 ? rawRequestId : undefined

  return new LlmError(message, code, {
    status,
    ...retryAfterMs === undefined ? {} : { providerRetryAfterMs: retryAfterMs },
    ...requestId === undefined ? {} : { requestId: ProviderRequestId(requestId) },
  })
}

/**
 * Incrementally decode a `text/event-stream` byte stream into {@link SseEvent}s.
 */
export async function* parseSseEvents(
  body: AsyncIterable<Uint8Array>,
  onActivity?: () => void,
): AsyncGenerator<SseEvent> {
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let eventName: string | undefined
  let dataLines: string[] = []

  function flushEvent(): SseEvent | undefined {
    if (dataLines.length === 0) {
      eventName = undefined
      return undefined
    }
    const event: SseEvent = {
      ...eventName === undefined ? {} : { event: eventName },
      data: dataLines.join('\n'),
    }
    eventName = undefined
    dataLines = []
    return event
  }

  function* processLine(rawLine: string): Generator<SseEvent> {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine
    if (line.length === 0) {
      const dispatched = flushEvent()
      if (dispatched !== undefined) yield dispatched
      return
    }
    if (line.startsWith(':')) return
    const colon = line.indexOf(':')
    const field = colon === -1 ? line : line.slice(0, colon)
    const rawValue = colon === -1 ? '' : line.slice(colon + 1)
    const value = rawValue.startsWith(' ') ? rawValue.slice(1) : rawValue
    if (field === 'event') {
      eventName = value
    } else if (field === 'data') {
      dataLines.push(value)
    }
  }

  for await (const chunk of body) {
    onActivity?.()
    buffer += decoder.decode(chunk, { stream: true })
    let newline = buffer.indexOf('\n')
    while (newline !== -1) {
      const line = buffer.slice(0, newline)
      buffer = buffer.slice(newline + 1)
      yield* processLine(line)
      newline = buffer.indexOf('\n')
    }
  }

  buffer += decoder.decode()
  if (buffer.length > 0) {
    for (const line of buffer.split('\n')) {
      yield* processLine(line)
    }
  }
  const trailing = flushEvent()
  if (trailing !== undefined) yield trailing
}

interface OpenBlock {
  index: number
  kind: 'text' | 'reasoning' | 'tool-call'
  text: string
  callId?: string
  name?: string
  sawDelta?: boolean
}

function closeBlock(block: OpenBlock): ContentBlock {
  switch (block.kind) {
    case 'text': return { type: 'text', text: block.text }
    case 'reasoning': return { type: 'reasoning', text: block.text }
    case 'tool-call': return {
      type: 'tool-call',
      id: ToolCallId(block.callId ?? ''),
      name: block.name ?? '',
      arguments: block.text,
    }
  }
}

function usageFromResponses(rawUsage: unknown): TokenUsage | undefined {
  const usage = asRecord(rawUsage)
  if (usage === undefined) return undefined
  const inputDetails = asRecord(usage.input_tokens_details)
  const outputDetails = asRecord(usage.output_tokens_details)

  const rawInput = nonNegativeInt(usage.input_tokens)
  const cachedFromDetails = nonNegativeInt(inputDetails?.cached_tokens)
  const cachedFromAnthropic = nonNegativeInt(usage.cache_read_input_tokens)
  const cacheReadTokens = cachedFromDetails ?? cachedFromAnthropic ?? 0
  const cacheWriteTokens = nonNegativeInt(usage.cache_creation_input_tokens) ?? 0
  const outputTokens = nonNegativeInt(usage.output_tokens) ?? 0
  const reasoningTokens = nonNegativeInt(outputDetails?.reasoning_tokens)
  const total = nonNegativeInt(usage.total_tokens)

  const inputTokens = rawInput === undefined
    ? 0
    : cachedFromDetails !== undefined
      ? Math.max(0, rawInput - cachedFromDetails)
      : rawInput

  if (inputTokens === 0 && outputTokens === 0 && cacheReadTokens === 0 && cacheWriteTokens === 0) {
    return undefined
  }
  const bucketSum = inputTokens + cacheReadTokens + cacheWriteTokens + outputTokens
  return {
    inputTokens,
    outputTokens,
    ...total !== undefined && total >= bucketSum ? { totalTokens: total } : {},
    ...cacheReadTokens > 0 ? { cacheReadTokens } : {},
    ...cacheWriteTokens > 0 ? { cacheWriteTokens } : {},
    ...reasoningTokens !== undefined && reasoningTokens > 0 ? { reasoningTokens } : {},
  }
}

function finishReasonFromResponse(
  response: Record<string, unknown> | undefined,
  sawToolCalls: boolean,
): FinishReason {
  const status = typeof response?.status === 'string' ? response.status : 'completed'
  if (status === 'incomplete') {
    const details = asRecord(response?.incomplete_details)
    const reason = typeof details?.reason === 'string' ? details.reason : undefined
    if (reason === 'content_filter') {
      return {
        kind: 'error',
        failure: {
          message: 'Kiro stopped the turn with a content filter',
          code: 'CONTENT_FILTERED',
        },
      }
    }
    return { kind: 'max-tokens' }
  }
  return sawToolCalls ? { kind: 'tool-calls' } : { kind: 'stop' }
}

/**
 * Translate a `/v1/responses` SSE byte stream into DSH {@link StreamChunk}s.
 *
 * Emits `block-start`, delta chunks (`text-delta`, `reasoning-delta`,
 * `tool-call-delta`), `block-end`, `usage`, and terminal `finish`. Throws
 * {@link LlmError} on SSE error events, malformed JSON, or truncated streams
 * that end before `response.completed` / `response.incomplete`.
 */
export async function* translateCreateResponseSse(
  body: AsyncIterable<Uint8Array>,
  onActivity?: () => void,
): AsyncGenerator<StreamChunk> {
  const order: OpenBlock[] = []
  const byItemId = new Map<string, OpenBlock>()
  const byOutputIndex = new Map<number, OpenBlock>()
  let nextIndex = 0
  let textBlock: OpenBlock | undefined
  let reasoningBlock: OpenBlock | undefined
  let toolCount = 0
  let usage: TokenUsage | undefined
  let completed = false
  let terminalResponse: Record<string, unknown> | undefined

  function openBlock(kind: OpenBlock['kind'], itemId?: string, outputIndex?: number): OpenBlock {
    const block: OpenBlock = { index: nextIndex++, kind, text: '' }
    order.push(block)
    if (itemId !== undefined && itemId.length > 0) byItemId.set(itemId, block)
    if (outputIndex !== undefined) byOutputIndex.set(outputIndex, block)
    if (kind === 'tool-call') toolCount += 1
    return block
  }

  function lookupBlock(itemId?: string, outputIndex?: number): OpenBlock | undefined {
    if (itemId !== undefined && byItemId.has(itemId)) return byItemId.get(itemId)
    if (outputIndex !== undefined && byOutputIndex.has(outputIndex)) return byOutputIndex.get(outputIndex)
    return undefined
  }

  function* emitText(text: string, itemId?: string, outputIndex?: number, isDelta = true): Generator<StreamChunk> {
    if (text.length === 0) return
    let block = lookupBlock(itemId, outputIndex)
    if (block === undefined || block.kind !== 'text') {
      const hasIdentity = (itemId !== undefined && itemId.length > 0) || outputIndex !== undefined
      block = hasIdentity
        ? openBlock('text', itemId, outputIndex)
        : (textBlock ?? openBlock('text', itemId, outputIndex))
      if (textBlock === undefined || hasIdentity) {
        textBlock = block
        yield { type: 'block-start', index: block.index, blockType: 'text' }
      }
      if (itemId !== undefined && itemId.length > 0) byItemId.set(itemId, block)
      if (outputIndex !== undefined) byOutputIndex.set(outputIndex, block)
    }
    if (isDelta) block.sawDelta = true
    block.text += text
    yield { type: 'text-delta', index: block.index, text }
  }

  function* emitReasoning(text: string, itemId?: string, outputIndex?: number, isDelta = true): Generator<StreamChunk> {
    if (text.length === 0) return
    let block = lookupBlock(itemId, outputIndex)
    if (block === undefined || block.kind !== 'reasoning') {
      const hasIdentity = (itemId !== undefined && itemId.length > 0) || outputIndex !== undefined
      block = hasIdentity
        ? openBlock('reasoning', itemId, outputIndex)
        : (reasoningBlock ?? openBlock('reasoning', itemId, outputIndex))
      if (reasoningBlock === undefined || hasIdentity) {
        reasoningBlock = block
        yield { type: 'block-start', index: block.index, blockType: 'reasoning' }
      }
      if (itemId !== undefined && itemId.length > 0) byItemId.set(itemId, block)
      if (outputIndex !== undefined) byOutputIndex.set(outputIndex, block)
    }
    if (isDelta) block.sawDelta = true
    block.text += text
    yield { type: 'reasoning-delta', index: block.index, text }
  }

  for await (const sse of parseSseEvents(body, onActivity)) {
    if (sse.data.trim() === '[DONE]') {
      completed = true
      break
    }
    let parsed: Record<string, unknown>
    try {
      const raw = JSON.parse(sse.data) as unknown
      const obj = asRecord(raw)
      if (obj === undefined) {
        throw new LlmError('malformed Kiro CreateResponse SSE event payload', 'MALFORMED_RESPONSE')
      }
      parsed = obj
    } catch (error: unknown) {
      if (error instanceof LlmError) throw error
      throw new LlmError(
        `malformed Kiro CreateResponse SSE event payload: ${sse.data.slice(0, 120)}`,
        'MALFORMED_RESPONSE',
        { cause: error },
      )
    }

    const eventType = typeof parsed.type === 'string' ? parsed.type : sse.event ?? ''
    const itemId = typeof parsed.item_id === 'string' ? parsed.item_id : undefined
    const outputIndex = typeof parsed.output_index === 'number' ? parsed.output_index : undefined

    if (eventType === 'error' || eventType === 'response.failed') {
      const resp = asRecord(parsed.response)
      const errObj = asRecord(resp?.error) ?? asRecord(parsed.error) ?? parsed
      const message = typeof errObj.message === 'string' && errObj.message.length > 0
        ? errObj.message
        : 'Kiro CreateResponse stream failed'
      const code = mapCreateResponseErrorCode(
        500,
        typeof errObj.type === 'string' ? errObj.type : undefined,
        typeof errObj.code === 'string' ? errObj.code : undefined,
        message,
        sse.data,
      )
      throw new LlmError(message, code)
    }

    if (eventType === 'response.output_item.added') {
      const item = asRecord(parsed.item)
      const id = typeof item?.id === 'string' ? item.id : itemId
      if (item?.type === 'function_call') {
        const callId = typeof item.call_id === 'string' && item.call_id.length > 0
          ? item.call_id
          : id ?? ''
        const name = typeof item.name === 'string' ? item.name : undefined
        const block = openBlock('tool-call', id, outputIndex)
        block.callId = callId
        if (name !== undefined && name.length > 0) block.name = name
        yield { type: 'block-start', index: block.index, blockType: 'tool-call' }
        const initialArgs = typeof item.arguments === 'string' ? item.arguments : ''
        if (initialArgs.length > 0) block.sawDelta = true
        if ((block.name !== undefined && block.name.length > 0) || initialArgs.length > 0) {
          block.text += initialArgs
          yield {
            type: 'tool-call-delta',
            index: block.index,
            id: ToolCallId(callId),
            ...block.name === undefined ? {} : { name: block.name },
            argumentsDelta: initialArgs,
          }
        }
      }
      // Text and reasoning blocks open lazily on their first non-empty delta so
      // an empty item or early stream interruption emits no empty block-start.
      continue
    }

    if (eventType === 'response.output_text.delta') {
      const delta = typeof parsed.delta === 'string' ? parsed.delta : ''
      yield* emitText(delta, itemId, outputIndex, true)
      continue
    }

    if (eventType === 'response.output_text.done') {
      const block = lookupBlock(itemId, outputIndex) ?? textBlock
      const fullText = typeof parsed.text === 'string' ? parsed.text : ''
      if (fullText.length > 0 && block?.sawDelta !== true) {
        yield* emitText(fullText, itemId, outputIndex, false)
      }
      continue
    }

    if (eventType.startsWith('response.reasoning') && eventType.endsWith('.delta')) {
      const delta = typeof parsed.delta === 'string'
        ? parsed.delta
        : typeof parsed.text === 'string' ? parsed.text : ''
      yield* emitReasoning(delta, itemId, outputIndex, true)
      continue
    }

    if (eventType.startsWith('response.reasoning') && eventType.endsWith('.done')) {
      const block = lookupBlock(itemId, outputIndex) ?? reasoningBlock
      const fullText = typeof parsed.text === 'string' ? parsed.text : ''
      if (fullText.length > 0 && block?.sawDelta !== true) {
        yield* emitReasoning(fullText, itemId, outputIndex, false)
      }
      continue
    }

    if (eventType === 'response.function_call_arguments.delta') {
      const delta = typeof parsed.delta === 'string' ? parsed.delta : ''
      let block = lookupBlock(itemId, outputIndex)
      if (block === undefined || block.kind !== 'tool-call') {
        block = openBlock('tool-call', itemId, outputIndex)
        block.callId = itemId ?? ''
        yield { type: 'block-start', index: block.index, blockType: 'tool-call' }
      }
      if (delta.length > 0) block.sawDelta = true
      block.text += delta
      yield {
        type: 'tool-call-delta',
        index: block.index,
        id: ToolCallId(block.callId ?? ''),
        ...block.name === undefined ? {} : { name: block.name },
        argumentsDelta: delta,
      }
      continue
    }

    if (eventType === 'response.function_call_arguments.done') {
      const block = lookupBlock(itemId, outputIndex)
      const args = typeof parsed.arguments === 'string' ? parsed.arguments : ''
      if (block !== undefined && block.kind === 'tool-call' && block.sawDelta !== true && args.length > 0) {
        block.sawDelta = true
        block.text += args
        yield {
          type: 'tool-call-delta',
          index: block.index,
          id: ToolCallId(block.callId ?? ''),
          ...block.name === undefined ? {} : { name: block.name },
          argumentsDelta: args,
        }
      }
      continue
    }

    if (eventType === 'response.output_item.done') {
      const item = asRecord(parsed.item)
      const id = typeof item?.id === 'string' ? item.id : itemId
      if (item?.type === 'function_call') {
        let block = lookupBlock(id, outputIndex)
        const callId = typeof item.call_id === 'string' && item.call_id.length > 0 ? item.call_id : id ?? ''
        const name = typeof item.name === 'string' && item.name.length > 0 ? item.name : undefined
        if (block === undefined || block.kind !== 'tool-call') {
          block = openBlock('tool-call', id, outputIndex)
          block.callId = callId
          if (name !== undefined) block.name = name
          yield { type: 'block-start', index: block.index, blockType: 'tool-call' }
        } else {
          if ((block.callId === undefined || block.callId.length === 0) && callId.length > 0) block.callId = callId
          if (block.name === undefined && name !== undefined) block.name = name
        }
        const args = typeof item.arguments === 'string' ? item.arguments : ''
        if (block.sawDelta !== true && args.length > 0) {
          block.sawDelta = true
          block.text += args
          yield {
            type: 'tool-call-delta',
            index: block.index,
            id: ToolCallId(block.callId ?? ''),
            ...block.name === undefined ? {} : { name: block.name },
            argumentsDelta: args,
          }
        }
      }
      continue
    }

    if (eventType === 'response.completed' || eventType === 'response.incomplete') {
      const response = asRecord(parsed.response)
      if (response?.status === 'failed') {
        const errObj = asRecord(response.error)
        const message = typeof errObj?.message === 'string' && errObj.message.length > 0
          ? errObj.message
          : 'Kiro CreateResponse stream failed'
        throw new LlmError(
          message,
          mapCreateResponseErrorCode(
            500,
            typeof errObj?.type === 'string' ? errObj.type : undefined,
            typeof errObj?.code === 'string' ? errObj.code : undefined,
            message,
            sse.data,
          ),
        )
      }
      usage = usageFromResponses(response?.usage) ?? usage
      terminalResponse = response
      completed = true
      break
    }
  }

  if (!completed) {
    throw new LlmError(
      'Kiro CreateResponse stream ended before completion',
      order.length === 0 ? EMPTY_RESPONSE_CODE : 'TRANSPORT',
    )
  }

  for (const block of order) {
    yield { type: 'block-end', index: block.index, block: closeBlock(block) }
  }

  const finish = finishReasonFromResponse(terminalResponse, toolCount > 0)
  const settled: FinishReason = order.length === 0 && (finish.kind === 'stop' || finish.kind === 'tool-calls')
    ? {
      kind: 'error',
      failure: {
        message: 'Kiro returned a completed response with no content',
        code: EMPTY_RESPONSE_CODE,
      },
    }
    : finish

  if (usage !== undefined) yield { type: 'usage', usage }
  yield { type: 'finish', reason: settled }
}

/**
 * Send one streaming request to `POST /v1/responses` and translate its SSE stream.
 */
export async function* streamCreateResponse(
  params: StreamCreateResponseParams,
): AsyncGenerator<StreamChunk> {
  const postRequest = params.postRequest ?? defaultPost()
  const payload = convertWireRequestToCreateResponse(params.wireRequest)
  const response = await postRequest({
    url: createResponseEndpoint(params.region),
    headers: createResponseHeaders(params.token, params.profileArn),
    body: JSON.stringify(payload),
    signal: params.signal,
    ...params.proxyUrl === undefined ? {} : { proxyUrl: params.proxyUrl },
  })

  if (response.status !== 200) {
    const chunks: Uint8Array[] = []
    for await (const chunk of response.body) chunks.push(chunk)
    const text = Buffer.concat(chunks).toString('utf8')
    throw parseCreateResponseError(response.status, text, response.headers)
  }

  yield* translateCreateResponseSse(response.body, params.onActivity)
}
