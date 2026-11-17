import { beforeEach, describe, expect, it } from 'vitest'
import {
  BlockAssembler,
  createAssistantMessage,
  createToolResultMessage,
  createUserMessage,
  LlmError,
  ToolCallId,
} from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, Message, StreamChunk } from '@deepseek-ai/dsh-llm'
import { KiroAdapter } from '../src/adapter.ts'
import type { KiroConnectionOptions } from '../src/adapter.ts'
import type { KiroToken } from '../src/auth.ts'
import { KIRO_CLIENT_VERSION } from '../src/client-id.ts'
import {
  clearFeatureConfigCache,
  convertWireRequestToCreateResponse,
  createResponseEndpoint,
  createResponseHeaders,
  createResponseTokenTypeHeaders,
  FEATURE_CONFIG_FAILURE_TTL_MS,
  FEATURE_CONFIG_TTL_MS,
  featureConfigEndpoint,
  parseCreateResponseError,
  probeFeatureConfiguration,
  translateCreateResponseSse,
} from '../src/create-response.ts'
import { serializeRequest } from '../src/serialize.ts'
import type { TransportRequest, TransportResponse } from '../src/transport.ts'
import { frame, join, metadataFrame, textFrame } from './frames.ts'

const SOURCE = { kind: 'plugin' as const, plugin: 'test' }
const PROFILE_ARN = 'arn:aws:codewhisperer:us-east-1:123456789012:profile/TEST'

function user(text: string): Message {
  return createUserMessage({ content: [{ type: 'text', text }], source: SOURCE })
}

function assistant(text: string, calls: { id: string; name: string; args: string }[] = []): Message {
  return createAssistantMessage({
    content: [
      ...text.length > 0 ? [{ type: 'text' as const, text }] : [],
      ...calls.map(call => ({
        type: 'tool-call' as const,
        id: ToolCallId(call.id),
        name: call.name,
        arguments: call.args,
      })),
    ],
    source: { provider: 'kiro', model: 'claude-sonnet-4.5' },
  })
}

function toolResult(id: string, text: string, isError = false): Message {
  return createToolResultMessage({
    callId: ToolCallId(id),
    content: [{ type: 'text', text }],
    isError,
  })
}

function makeToken(overrides: Partial<KiroToken> = {}): KiroToken {
  return {
    accessToken: 'tok-builder-1',
    region: 'us-east-1',
    expiresAt: Date.now() + 60_000,
    authMethod: 'builder-id',
    profileArn: PROFILE_ARN,
    accountId: 'acc-1',
    ...overrides,
  }
}

function makeConnection(overrides: Partial<KiroConnectionOptions> = {}): KiroConnectionOptions {
  return {
    region: 'us-east-1',
    profileArn: PROFILE_ARN,
    defaults: {},
    defaultContextWindow: 200_000,
    models: [{
      id: 'claude-sonnet-4.5',
      name: 'Claude Sonnet 4.5',
      thinking: true,
      inputModalities: ['text', 'image'],
    }],
    streamIdleTimeoutMs: 300_000,
    tokenExpiryBufferMs: 300_000,
    retryPolicy: {},
    ...overrides,
  }
}

function sseBytes(events: { event?: string; data: unknown }[], crlf = false): Uint8Array {
  const sep = crlf ? '\r\n' : '\n'
  const text = events
    .map((evt) => {
      const data = typeof evt.data === 'string' ? evt.data : JSON.stringify(evt.data)
      return evt.event === undefined
        ? `data: ${data}${sep}${sep}`
        : `event: ${evt.event}${sep}data: ${data}${sep}${sep}`
    })
    .join('')
  return new TextEncoder().encode(text)
}

async function* chunkBytes(bytes: Uint8Array, chunkSize: number): AsyncGenerator<Uint8Array> {
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    yield bytes.slice(offset, offset + chunkSize)
  }
}

async function collectChunks(stream: AsyncIterable<StreamChunk>): Promise<StreamChunk[]> {
  const chunks: StreamChunk[] = []
  for await (const chunk of stream) chunks.push(chunk)
  return chunks
}

interface RecordedGet {
  url: string
  headers: Record<string, string>
  proxyUrl: string | undefined
}

interface RecordedPost {
  url: string
  headers: Record<string, string>
  body: string
  proxyUrl?: string
}

describe('CreateResponse (a) SSE parser and error normalization', () => {
  it('parses reasoning, text, tool calls, and disjoint usage across arbitrary chunk boundaries', async () => {
    const raw = sseBytes([
      { event: 'response.created', data: { type: 'response.created', response: { id: 'resp_1', status: 'in_progress' } } },
      { event: 'response.in_progress', data: { type: 'response.in_progress', response: { id: 'resp_1', status: 'in_progress' } } },
      { event: 'response.output_item.added', data: { type: 'response.output_item.added', output_index: 0, item: { id: 'rs_1', type: 'reasoning', summary: [] } } },
      { event: 'response.reasoning_summary_text.delta', data: { type: 'response.reasoning_summary_text.delta', item_id: 'rs_1', output_index: 0, delta: 'Step 1. ' } },
      { event: 'response.reasoning.delta', data: { type: 'response.reasoning.delta', item_id: 'rs_1', output_index: 0, delta: 'Step 2.' } },
      { event: 'response.output_item.done', data: { type: 'response.output_item.done', output_index: 0, item: { id: 'rs_1', type: 'reasoning' } } },
      { event: 'response.output_item.added', data: { type: 'response.output_item.added', output_index: 1, item: { id: 'msg_1', type: 'message', role: 'assistant', content: [] } } },
      { event: 'response.content_part.added', data: { type: 'response.content_part.added', item_id: 'msg_1', output_index: 1, content_index: 0, part: { type: 'output_text', text: '' } } },
      { event: 'response.output_text.delta', data: { type: 'response.output_text.delta', item_id: 'msg_1', output_index: 1, content_index: 0, delta: '你好，' } },
      { event: 'response.output_text.delta', data: { type: 'response.output_text.delta', item_id: 'msg_1', output_index: 1, content_index: 0, delta: 'world!' } },
      { event: 'response.output_text.done', data: { type: 'response.output_text.done', item_id: 'msg_1', output_index: 1, content_index: 0, text: '你好，world!' } },
      { event: 'response.output_item.added', data: { type: 'response.output_item.added', output_index: 2, item: { id: 'fc_1', type: 'function_call', call_id: 'call_1', name: 'read_file', arguments: '' } } },
      { event: 'response.function_call_arguments.delta', data: { type: 'response.function_call_arguments.delta', item_id: 'fc_1', output_index: 2, delta: '{"path":' } },
      { event: 'response.function_call_arguments.delta', data: { type: 'response.function_call_arguments.delta', item_id: 'fc_1', output_index: 2, delta: '"src/index.ts"}' } },
      { event: 'response.function_call_arguments.done', data: { type: 'response.function_call_arguments.done', item_id: 'fc_1', output_index: 2, arguments: '{"path":"src/index.ts"}' } },
      { event: 'response.output_item.done', data: { type: 'response.output_item.done', output_index: 2, item: { id: 'fc_1', type: 'function_call', call_id: 'call_1', name: 'read_file', arguments: '{"path":"src/index.ts"}' } } },
      {
        event: 'response.completed',
        data: {
          type: 'response.completed',
          response: {
            id: 'resp_1',
            status: 'completed',
            usage: {
              input_tokens: 150,
              input_tokens_details: { cached_tokens: 50 },
              output_tokens: 40,
              output_tokens_details: { reasoning_tokens: 12 },
              total_tokens: 190,
            },
          },
        },
      },
    ], true)

    // Use 7-byte chunks to split multi-byte UTF-8 characters and CRLF lines.
    const chunks = await collectChunks(translateCreateResponseSse(chunkBytes(raw, 7)))
    const assembler = new BlockAssembler()
    for (const chunk of chunks) assembler.push(chunk)

    expect(assembler.blocks()).toEqual([
      { type: 'reasoning', text: 'Step 1. Step 2.' },
      { type: 'text', text: '你好，world!' },
      { type: 'tool-call', id: 'call_1', name: 'read_file', arguments: '{"path":"src/index.ts"}' },
    ])
    expect(assembler.usage).toEqual({
      inputTokens: 100,
      outputTokens: 40,
      totalTokens: 190,
      cacheReadTokens: 50,
      reasoningTokens: 12,
    })
    expect(assembler.finish).toEqual({ kind: 'tool-calls' })
  })

  it('derives stop for text-only completion and max-tokens / content_filter for incomplete', async () => {
    const textOnly = sseBytes([
      { event: 'response.output_text.delta', data: { type: 'response.output_text.delta', delta: 'hello' } },
      { event: 'response.completed', data: { type: 'response.completed', response: { status: 'completed' } } },
    ])
    const textChunks = await collectChunks(translateCreateResponseSse(chunkBytes(textOnly, 64)))
    expect(textChunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'stop' } })

    const truncated = sseBytes([
      { event: 'response.output_text.delta', data: { type: 'response.output_text.delta', delta: 'cut' } },
      { event: 'response.incomplete', data: { type: 'response.incomplete', response: { status: 'incomplete', incomplete_details: { reason: 'max_output_tokens' } } } },
    ])
    const truncChunks = await collectChunks(translateCreateResponseSse(chunkBytes(truncated, 64)))
    expect(truncChunks.at(-1)).toEqual({ type: 'finish', reason: { kind: 'max-tokens' } })

    const filtered = sseBytes([
      { event: 'response.output_text.delta', data: { type: 'response.output_text.delta', delta: 'partial' } },
      { event: 'response.incomplete', data: { type: 'response.incomplete', response: { status: 'incomplete', incomplete_details: { reason: 'content_filter' } } } },
    ])
    const filteredChunks = await collectChunks(translateCreateResponseSse(chunkBytes(filtered, 64)))
    expect(filteredChunks.at(-1)).toEqual({
      type: 'finish',
      reason: {
        kind: 'error',
        failure: { message: 'Kiro stopped the turn with a content filter', code: 'CONTENT_FILTERED' },
      },
    })
  })

  it('throws LlmError on truncated SSE streams or in-band error events', async () => {
    const interrupted = sseBytes([
      { event: 'response.created', data: { type: 'response.created', response: { id: 'resp_x' } } },
    ])
    await expect(collectChunks(translateCreateResponseSse(chunkBytes(interrupted, 64))))
      .rejects.toMatchObject({ code: 'EMPTY_RESPONSE' })

    const midCut = sseBytes([
      { event: 'response.output_text.delta', data: { type: 'response.output_text.delta', delta: 'half' } },
    ])
    await expect(collectChunks(translateCreateResponseSse(chunkBytes(midCut, 64))))
      .rejects.toMatchObject({ code: 'TRANSPORT' })

    const failedEvt = sseBytes([
      { event: 'error', data: { type: 'error', error: { message: 'Overloaded', type: 'server_is_overloaded', code: 'server_is_overloaded' } } },
    ])
    await expect(collectChunks(translateCreateResponseSse(chunkBytes(failedEvt, 64))))
      .rejects.toMatchObject({ code: 'SERVER', message: 'Overloaded' })
  })

  it('normalizes non-2xx JSON errors including retryAfterMilliseconds', () => {
    const rateLimit = parseCreateResponseError(
      429,
      JSON.stringify({
        error: {
          message: 'Slow down',
          type: 'rate_limit_exceeded',
          param: null,
          code: 'rate_limit_exceeded',
        },
        retryAfterMilliseconds: 2500,
      }),
      { 'x-amzn-requestid': 'req-429' },
    )
    expect(rateLimit).toBeInstanceOf(LlmError)
    expect(rateLimit.code).toBe('RATE_LIMIT')
    expect(rateLimit.message).toBe('Slow down')
    expect(rateLimit.failure.status).toBe(429)
    expect(rateLimit.failure.providerRetryAfterMs).toBe(2500)
    expect(rateLimit.failure.requestId).toBe('req-429')

    const overloaded = parseCreateResponseError(
      503,
      JSON.stringify({
        error: {
          message: 'Server overloaded',
          type: 'server_is_overloaded',
          code: 'server_is_overloaded',
        },
      }),
    )
    expect(overloaded.code).toBe('SERVER')

    const overflow = parseCreateResponseError(
      400,
      JSON.stringify({
        error: {
          message: 'Prompt is too long',
          type: 'invalid_request',
          code: 'context_length_exceeded',
        },
      }),
    )
    expect(overflow.code).toBe('CONTEXT_WINDOW_EXCEEDED')
  })
})

describe('CreateResponse (b) request conversion and headers', () => {
  it('converts multi-turn history, tool calls, tool results, images, and tool schemas', () => {
    const images = new Map([
      ['img-1', { format: 'png' as const, source: { bytes: 'AQID' } }],
    ])
    const userImgMsg = createUserMessage({
      content: [
        { type: 'text', text: 'Look at this screenshot' },
        { type: 'image', attachment: { attachmentId: 'img-1', mediaType: 'image/png', bytes: 3, width: 1, height: 1 } as never },
      ],
      source: SOURCE,
    })

    const wire = serializeRequest(
      {
        provider: 'kiro',
        model: 'claude-sonnet-4.5',
        system: 'You are concise.',
        messages: [
          user('Check config.json'),
          assistant('', [{ id: 'call_1', name: 'read_file', args: '{"path":"config.json"}' }]),
          toolResult('call_1', '{"ok":true}'),
          userImgMsg,
        ],
        tools: [{
          name: 'read_file',
          description: 'Read a workspace file',
          parameters: {
            type: 'object',
            properties: { path: { type: 'string' } },
            required: ['path'],
          },
        }],
      },
      {},
      'conv-1',
      PROFILE_ARN,
      undefined,
      undefined,
      images,
    )

    const converted = convertWireRequestToCreateResponse(wire)
    expect(converted).toEqual({
      model: 'claude-sonnet-4.5',
      stream: true,
      profileArn: PROFILE_ARN,
      input: [
        {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text: 'You are concise.\n\nCheck config.json' }],
        },
        {
          type: 'function_call',
          call_id: 'call_1',
          name: 'read_file',
          arguments: '{"path":"config.json"}',
        },
        {
          type: 'function_call_output',
          call_id: 'call_1',
          output: '{"ok":true}',
        },
        {
          type: 'message',
          role: 'user',
          content: [
            { type: 'input_text', text: 'Look at this screenshot' },
            { type: 'input_image', image_url: 'data:image/png;base64,AQID' },
          ],
        },
      ],
      tools: [{
        type: 'function',
        name: 'read_file',
        description: 'Read a workspace file',
        parameters: {
          type: 'object',
          properties: { path: { type: 'string' } },
          required: ['path'],
        },
      }],
    })
  })

  it('builds the live-verified headers and TokenType per auth method', () => {
    const builderHeaders = createResponseHeaders(makeToken({ authMethod: 'builder-id' }), PROFILE_ARN)
    expect(builderHeaders).toEqual({
      authorization: 'Bearer tok-builder-1',
      'content-type': 'application/json',
      accept: 'text/event-stream',
      'user-agent': `aws-sdk-js/3.738.0 KiroIDE-${KIRO_CLIENT_VERSION}`,
      'x-amz-user-agent': `aws-sdk-js/3.738.0 KiroIDE-${KIRO_CLIENT_VERSION}`,
      'x-amzn-kiro-origin': 'AI_EDITOR',
      'x-amzn-kiro-agent-mode': 'vibe',
      'x-amzn-kiro-profile': PROFILE_ARN,
      'x-amzn-kiro-profile-arn': PROFILE_ARN,
      TokenType: 'SSO_OIDC',
    })
    expect(createResponseTokenTypeHeaders(makeToken({ authMethod: 'idc' }))).toEqual({ TokenType: 'SSO_OIDC' })
    expect(createResponseTokenTypeHeaders(makeToken({ authMethod: 'api_key' }))).toEqual({ TokenType: 'API_KEY' })
    expect(createResponseTokenTypeHeaders(makeToken({ authMethod: 'external_idp' }))).toEqual({ TokenType: 'EXTERNAL_IDP' })
  })
})

describe('CreateResponse (c) probe cache, mode switch, and fallback', () => {
  beforeEach(() => {
    clearFeatureConfigCache()
  })

  it('caches positive featureConfiguration probes for 15 minutes and isolates by accountId+profileArn+region', async () => {
    let clock = 1_000_000
    const recorded: RecordedGet[] = []
    const requestJson = async (url: string, headers: Record<string, string>, proxyUrl: string | undefined) => {
      recorded.push({ url, headers, proxyUrl })
      return { status: 200, body: { configuration: { abc123: true, def456: false } } }
    }

    const tokenA = makeToken({ accountId: 'acc-a' })
    const first = await probeFeatureConfiguration({
      token: tokenA,
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      requestJson,
      now: () => clock,
    })
    expect(first).toBe(true)
    expect(recorded).toHaveLength(1)
    expect(recorded[0]?.url).toBe(featureConfigEndpoint('us-east-1'))
    expect(recorded[0]?.headers['x-amzn-kiro-profile-arn']).toBe(PROFILE_ARN)

    // Still valid inside 15-minute window
    clock += FEATURE_CONFIG_TTL_MS - 1_000
    const second = await probeFeatureConfiguration({
      token: tokenA,
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      requestJson,
      now: () => clock,
    })
    expect(second).toBe(true)
    expect(recorded).toHaveLength(1)

    // Different account misses cache
    await probeFeatureConfiguration({
      token: makeToken({ accountId: 'acc-b' }),
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      requestJson,
      now: () => clock,
    })
    expect(recorded).toHaveLength(2)

    // Expired after 15 minutes
    clock += 2_000
    await probeFeatureConfiguration({
      token: tokenA,
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      requestJson,
      now: () => clock,
    })
    expect(recorded).toHaveLength(3)
  })

  it('caches failed probes for 60 seconds before retrying', async () => {
    let clock = 2_000_000
    let calls = 0
    const requestJson = async () => {
      calls += 1
      if (calls === 1) return { status: 500, body: { message: 'boom' } }
      return { status: 200, body: { configuration: { flag: true } } }
    }

    const token = makeToken({ accountId: 'acc-fail' })
    expect(await probeFeatureConfiguration({
      token,
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      requestJson,
      now: () => clock,
    })).toBe(false)
    expect(calls).toBe(1)

    clock += FEATURE_CONFIG_FAILURE_TTL_MS - 1_000
    expect(await probeFeatureConfiguration({
      token,
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      requestJson,
      now: () => clock,
    })).toBe(false)
    expect(calls).toBe(1)

    clock += 2_000
    expect(await probeFeatureConfiguration({
      token,
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      requestJson,
      now: () => clock,
    })).toBe(true)
    expect(calls).toBe(2)
  })

  it('exposes createResponseApi in Config schema with default auto and validates values', async () => {
    const { vi } = await import('vitest')
    vi.resetModules()
    vi.doMock('@deepseek-ai/cosmokit', () => ({
      isVolatile: (value: unknown) => typeof value === 'object' && value !== null && 'get' in value,
    }))
    const { Config, plainOptions, resolveAdapterOptions } = await import('../src/index.ts')
    expect(resolveAdapterOptions({}).createResponseApi).toBe('auto')
    expect(resolveAdapterOptions({ createResponseApi: 'on' }).createResponseApi).toBe('on')
    expect(resolveAdapterOptions({ createResponseApi: 'off' }).createResponseApi).toBe('off')
    expect(() => resolveAdapterOptions({ createResponseApi: 'invalid' as never })).toThrow(/createResponseApi/)

    const parsed = Config({})
    expect(plainOptions(parsed).createResponseApi).toBe('auto')
    vi.doUnmock('@deepseek-ai/cosmokit')
    vi.resetModules()
  })

  it('does not cache failure when caller aborts during probe', async () => {
    const controller = new AbortController()
    controller.abort('user cancelled')
    let calls = 0
    await expect(probeFeatureConfiguration({
      token: makeToken({ accountId: 'acc-abort' }),
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      signal: controller.signal,
      requestJson: async () => {
        calls += 1
        throw new Error('aborted')
      },
    })).rejects.toMatchObject({ code: 'ABORTED' })
    expect(calls).toBe(1)

    // Subsequent call with a live signal still probes (nothing cached on abort).
    const ok = await probeFeatureConfiguration({
      token: makeToken({ accountId: 'acc-abort' }),
      profileArn: PROFILE_ARN,
      region: 'us-east-1',
      requestJson: async () => {
        calls += 1
        return { status: 200, body: { configuration: { enabled: true } } }
      },
    })
    expect(ok).toBe(true)
    expect(calls).toBe(2)
  })
})

describe('CreateResponse (c) adapter routing & fallback with fake transport', () => {
  beforeEach(() => {
    clearFeatureConfigCache()
  })

  it('routes auto (probe=true) -> /v1/responses and auto (non-2xx / stream interrupt) -> legacy fallback with debug log', async () => {
    const { vi } = await import('vitest')
    vi.resetModules()

    const gets: RecordedGet[] = []
    const posts: RecordedPost[] = []
    const getQueue: (() => Promise<{ status: number; body: unknown }>)[] = []
    const postQueue: (() => Promise<TransportResponse>)[] = []

    vi.doMock('../src/transport.ts', () => ({
      getJson: async (url: string, headers: Record<string, string>, proxyUrl: string | undefined) => {
        gets.push({ url, headers, proxyUrl })
        const next = getQueue.shift()
        if (next === undefined) throw new Error(`unexpected GET ${url}`)
        return next()
      },
      post: async (req: TransportRequest) => {
        posts.push({ url: req.url, headers: req.headers, body: req.body })
        const next = postQueue.shift()
        if (next === undefined) throw new Error(`unexpected POST ${req.url}`)
        return next()
      },
      parseProxyUrl: (raw: string) => new URL(raw),
      postJson: async () => ({ status: 200, body: {} }),
      postJsonWithHeaders: async () => ({ status: 200, body: {} }),
      postForm: async () => ({ status: 200, body: {} }),
    }))

    const { KiroAdapter: MockedAdapter } = await import('../src/adapter.ts')
    const { clearFeatureConfigCache: clearCache } = await import('../src/create-response.ts')
    clearCache()

    const debugLogs: string[] = []
    const createAdapter = (mode?: 'auto' | 'on' | 'off') => new MockedAdapter({
      options: () => makeConnection(mode === undefined ? {} : { createResponseApi: mode }),
      resolveToken: async () => makeToken(),
      logDebug: (msg) => { debugLogs.push(msg) },
    })

    const runTurn = async (instance: InstanceType<typeof MockedAdapter>) => {
      return collectChunks(instance.stream({
        provider: 'kiro',
        model: 'claude-sonnet-4.5',
        messages: [user('hi')],
      } as GenerateOptions))
    }

    // 1. Auto mode with probe=true -> hits /v1/responses
    getQueue.push(async () => ({ status: 200, body: { configuration: { f1: true } } }))
    postQueue.push(async () => ({
      status: 200,
      headers: {},
      body: chunkBytes(sseBytes([
        { event: 'response.output_text.delta', data: { type: 'response.output_text.delta', delta: 'from-responses' } },
        { event: 'response.completed', data: { type: 'response.completed', response: { status: 'completed' } } },
      ]), 64),
    }))
    const res1 = await runTurn(createAdapter('auto'))
    expect(res1.some(c => c.type === 'text-delta' && c.text === 'from-responses')).toBe(true)
    expect(gets).toHaveLength(1)
    expect(posts).toHaveLength(1)
    expect(posts[0]?.url).toBe(createResponseEndpoint('us-east-1'))

    // 2. Auto mode with cached probe=true, /v1/responses fails with non-2xx -> falls back to legacy
    postQueue.push(
      async () => ({
        status: 500,
        headers: {},
        body: (async function* () {
          yield new TextEncoder().encode(JSON.stringify({
            error: { message: 'boom', type: 'server_is_overloaded', code: 'server_is_overloaded' },
          }))
        })(),
      }),
      async () => ({
        status: 200,
        headers: {},
        body: (async function* () {
          yield join(textFrame('from-legacy'), metadataFrame({ outputTokens: 2 }))
        })(),
      }),
    )
    const res2 = await runTurn(createAdapter('auto'))
    expect(res2.some(c => c.type === 'text-delta' && c.text === 'from-legacy')).toBe(true)
    expect(posts).toHaveLength(3)
    expect(posts[1]?.url).toBe(createResponseEndpoint('us-east-1'))
    expect(posts[2]?.url).toBe('https://q.us-east-1.amazonaws.com/generateAssistantResponse')
    expect(debugLogs).toHaveLength(1)
    expect(debugLogs[0]).toContain('falling back to generateAssistantResponse')
    expect(debugLogs[0]).toContain('boom')

    // 3. Auto mode with /v1/responses stream interruption before content -> falls back to legacy
    postQueue.push(
      async () => ({
        status: 200,
        headers: {},
        body: (async function* () {
          yield sseBytes([{ event: 'response.created', data: { type: 'response.created' } }])
          throw new Error('socket reset')
        })(),
      }),
      async () => ({
        status: 200,
        headers: {},
        body: (async function* () {
          yield join(textFrame('legacy-after-reset'), metadataFrame({ outputTokens: 2 }))
        })(),
      }),
    )
    const res3 = await runTurn(createAdapter('auto'))
    expect(res3.some(c => c.type === 'text-delta' && c.text === 'legacy-after-reset')).toBe(true)
    expect(debugLogs).toHaveLength(2)
    expect(debugLogs[1]).toContain('socket reset')

    // 4. Mode 'off' -> never probes, goes straight to legacy
    clearCache()
    postQueue.push(async () => ({
      status: 200,
      headers: {},
      body: (async function* () {
        yield join(textFrame('legacy-off'), metadataFrame({ outputTokens: 1 }))
      })(),
    }))
    const res4 = await runTurn(createAdapter('off'))
    expect(res4.some(c => c.type === 'text-delta' && c.text === 'legacy-off')).toBe(true)
    expect(gets).toHaveLength(1)
    expect(posts.at(-1)?.url).toBe('https://q.us-east-1.amazonaws.com/generateAssistantResponse')

    // 5. Mode 'on' -> skips probe, forces /v1/responses, throws normalized LlmError without fallback
    postQueue.push(async () => ({
      status: 429,
      headers: {},
      body: (async function* () {
        yield new TextEncoder().encode(JSON.stringify({
          error: { message: 'Rate limited', type: 'rate_limit_exceeded', code: 'rate_limit_exceeded' },
          retryAfterMilliseconds: 1200,
        }))
      })(),
    }))
    await expect(runTurn(createAdapter('on'))).rejects.toMatchObject({
      code: 'RATE_LIMIT',
      message: 'Rate limited',
      failure: expect.objectContaining({ providerRetryAfterMs: 1200, status: 429 }),
    })
    expect(gets).toHaveLength(1)

    vi.doUnmock('../src/transport.ts')
    vi.resetModules()
  })
})
