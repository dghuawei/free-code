/**
 * Integration tests for maybeSummarizeToolResultBlock with the sideQuery,
 * persistence, and config boundaries mocked. Verifies the fail-open contract:
 * every failure mode returns undefined so raw output enters context.
 *
 * mock.module is global in bun — afterAll restores to avoid leaking into
 * sibling test files.
 */
import { afterAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { mkdtemp, readFile, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const FIXED_CONFIG = {
  enabled: true,
  tokenThreshold: 2500,
  lineThreshold: 300,
  model: 'test-model',
  ignoredTools: new Set<string>(['Task']),
  maxInputChars: 200_000,
  timeoutMs: 30_000,
  summarizeServerToolResults: false,
  noThinkPromptSuffix: '',
  maxOutputTokens: 1024,
  streaming: false,
  liveViewFile: false,
}

mock.module('../../services/toolResultSummarization/config.js', () => ({
  DEFAULT_TOKEN_THRESHOLD: 2500,
  DEFAULT_LINE_THRESHOLD: 300,
  DEFAULT_MAX_INPUT_CHARS: 200_000,
  DEFAULT_TIMEOUT_MS: 30_000,
  getToolOutputSummarizationConfig: () => FIXED_CONFIG,
}))

type SideQueryResult = { content: Array<{ type: 'text'; text: string }> }

const sideQueryMock = mock(
  async (_opts: unknown): Promise<SideQueryResult> => {
    throw new Error('sideQuery mock not configured for this test')
  },
)

mock.module('../../utils/sideQuery.js', () => ({
  sideQuery: sideQueryMock,
}))

let liveDir = '/session/tool-results'

const persistMock = mock(async () => ({
  filepath: '/session/tool-results/toolu_1.txt',
  originalSize: 10_000,
  isJson: false,
  preview: '...',
  hasMore: true,
}))

mock.module('../../utils/toolResultStorage.js', () => ({
  hasImageBlock: (content: unknown) =>
    Array.isArray(content) &&
    content.some(
      (b: { type?: string }) => typeof b === 'object' && b.type === 'image',
    ),
  isContentAlreadyCompacted: (content: unknown) =>
    typeof content === 'string' &&
    (content.startsWith('<persisted-output>') ||
      content.startsWith('<tool-output-summary>')),
  isToolResultContentEmpty: (content: unknown) =>
    !content || (typeof content === 'string' && content.trim() === ''),
  persistToolResult: persistMock,
  getToolResultsDir: () => liveDir,
  PERSISTED_OUTPUT_TAG: '<persisted-output>',
  PERSISTED_OUTPUT_CLOSING_TAG: '</persisted-output>',
  TOOL_RESULTS_SUBDIR: 'tool-results',
  PREVIEW_SIZE_BYTES: 2000,
}))

const { maybeSummarizeToolResultBlock } = await import(
  './summarizeToolResult.js'
)

const LONG_OUTPUT = Array.from({ length: 400 }, (_, i) => `line ${i}`).join(
  '\n',
)

function baseParams() {
  return {
    toolResultBlock: {
      type: 'tool_result' as const,
      tool_use_id: 'toolu_1',
      content: LONG_OUTPUT,
      is_error: false,
    },
    toolName: 'Bash',
    maxResultSizeChars: 30_000,
    toolInput: { command: 'ls' },
    parentAbortController: new AbortController(),
    isSubagent: false,
    fallbackQuery: undefined as
      | ((args: {
          userPrompt: string
          systemPrompt: string
          model: string
          maxOutputTokens: number
          signal: AbortSignal
        }) => Promise<string>)
      | undefined,
  }
}

afterAll(() => {
  mock.restore()
})

beforeEach(() => {
  persistMock.mockClear()
  sideQueryMock.mockClear()
  sideQueryMock.mockImplementation(async (_opts: unknown) => {
    throw new Error('sideQuery mock not configured for this test')
  })
})

describe('maybeSummarizeToolResultBlock', () => {
  test('success returns a block whose content wraps the summary and file path', async () => {
    let seenOpts: Record<string, unknown> | undefined
    sideQueryMock.mockImplementationOnce(async (opts: unknown) => {
      seenOpts = opts as Record<string, unknown>
      return {
        content: [{ type: 'text', text: 'Listed 400 files; all under src/.' }],
      }
    })

    const result = await maybeSummarizeToolResultBlock(baseParams())

    expect(result).toBeDefined()
    expect(result!.type).toBe('tool_result')
    expect(result!.tool_use_id).toBe('toolu_1')
    expect(result!.is_error).toBe(false)
    expect(result!.content).toBe(
      '<tool-output-summary>\n' +
        'Listed 400 files; all under src/.\n' +
        '</tool-output-summary>\n' +
        'Full output (400 lines, 3.4KB) saved to: /session/tool-results/toolu_1.txt\n' +
        'Estimated tokens: 873 -> 67',
    )
    expect(persistMock).toHaveBeenCalledTimes(1)
    // Thinking must be explicitly disabled: reasoning models (qwen & co)
    // otherwise burn the whole max_tokens budget on reasoning and return
    // zero text blocks — the silent-skip failure mode observed in production.
    expect(seenOpts?.thinking).toBe(false)
  })

  test('summary copy is written next to the persisted raw output as <name>_sum.<ext>', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'sumcopy-'))
    const rawPath = join(dir, 'toolu_1.txt')
    await writeFile(rawPath, LONG_OUTPUT, 'utf-8')
    persistMock.mockImplementationOnce(async () => ({
      filepath: rawPath,
      originalSize: LONG_OUTPUT.length,
      isJson: false,
      preview: '...',
      hasMore: true,
    }))
    sideQueryMock.mockImplementationOnce(async () => ({
      content: [{ type: 'text', text: 'Concise digest of the 400 lines.' }],
    }))

    const result = await maybeSummarizeToolResultBlock(baseParams())

    expect(result).toBeDefined()
    const saved = await readFile(join(dir, 'toolu_1_sum.txt'), 'utf-8')
    expect(saved).toBe('Concise digest of the 400 lines.')
    // Raw output is left untouched — the copy is additive only
    expect(await readFile(rawPath, 'utf-8')).toBe(LONG_OUTPUT)
  })

  test('live view file: raw stage overwrite, then full rewrite with summary + token delta', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'liveview-'))
    liveDir = dir
    FIXED_CONFIG.liveViewFile = true
    const { writeLiveToolOutputView } = await import('./liveView.js')
    try {
      // Stage 1: raw write for EVERY tool result (toolExecution wiring);
      // a previous raw result is fully clobbered by the next one.
      await writeLiveToolOutputView({
        stage: 'raw',
        toolName: 'OtherTool',
        content: 'previous short output',
      })
      let live = await readFile(join(dir, 'live-view.txt'), 'utf-8')
      expect(live).toContain('OtherTool · RAW')
      expect(live).toContain('previous short output')

      // Stage 2: summarizer rewrite — header names the tool, the token
      // delta, and the file holds only the summary text.
      sideQueryMock.mockImplementationOnce(async () => ({
        content: [{ type: 'text', text: 'Digest of the raw output.' }],
      }))
      const result = await maybeSummarizeToolResultBlock(baseParams())
      expect(result).toBeDefined()
      live = await readFile(join(dir, 'live-view.txt'), 'utf-8')
      expect(live.startsWith('=== ')).toBe(true)
      expect(live).toContain('Bash · SUMMARY (~873 -> ~')
      expect(live).toContain('Digest of the raw output.')
      expect(live).not.toContain('previous short output')
    } finally {
      FIXED_CONFIG.liveViewFile = false
      liveDir = '/session/tool-results'
    }
  })

  test('reasoning-gateway workarounds: maxOutputTokens and streaming reach sideQuery', async () => {
    let seen: Record<string, unknown> | undefined
    sideQueryMock.mockImplementationOnce(async (opts: unknown) => {
      seen = opts as Record<string, unknown>
      return { content: [{ type: 'text', text: 'ok summary' }] }
    })
    FIXED_CONFIG.maxOutputTokens = 4096
    FIXED_CONFIG.streaming = true
    try {
      const result = await maybeSummarizeToolResultBlock(baseParams())
      expect(result).toBeDefined()
      expect(seen?.max_tokens).toBe(4096)
      expect(seen?.stream).toBe(true)
    } finally {
      FIXED_CONFIG.maxOutputTokens = 1024
      FIXED_CONFIG.streaming = false
    }
  })

  test('noThinkPromptSuffix is appended to the system prompt (Qwen3 soft switch)', async () => {
    let seenSystem: unknown
    sideQueryMock.mockImplementationOnce(async (opts: unknown) => {
      const o = opts as { system?: unknown }
      seenSystem = Array.isArray(o.system)
        ? (o.system as Array<{ type: string; text?: string }>)
            .map(b => (b.type === 'text' ? b.text : ''))
            .join('\n')
        : o.system
      return { content: [{ type: 'text', text: 'ok summary' }] }
    })
    FIXED_CONFIG.noThinkPromptSuffix = '/no_think'
    try {
      const params = baseParams()
      const result = await maybeSummarizeToolResultBlock(params)
      expect(result).toBeDefined()
      expect(String(seenSystem)).toContain('compress tool outputs')
      expect(String(seenSystem)).toEndWith('/no_think')
    } finally {
      FIXED_CONFIG.noThinkPromptSuffix = ''
    }
  })

  test('sideQuery failure falls back to main transport; both failing returns undefined', async () => {
    sideQueryMock.mockRejectedValueOnce(new Error('429 rate limited'))
    const params = baseParams()
    params.fallbackQuery = async () => ''

    expect(await maybeSummarizeToolResultBlock(params)).toBeUndefined()
  })

  test('empty sideQuery text + successful fallback yields a summary', async () => {
    // The production failure mode behind OpenAI-compatible gateways: the
    // direct Anthropic-protocol call returns 200 with empty text; the
    // fallback through the main transport recovers.
    sideQueryMock.mockResolvedValueOnce({ content: [{ type: 'text', text: '' }] })
    let seenFallbackArgs: {
      userPrompt: string
      systemPrompt: string
      model: string
      maxOutputTokens: number
    } | undefined
    const params = baseParams()
    params.fallbackQuery = async args => {
      seenFallbackArgs = args
      return 'Fallback summary: command printed 400 lines successfully.'
    }
    const result = await maybeSummarizeToolResultBlock(params)
    expect(result).toBeDefined()
    expect(result!.content).toContain('Fallback summary: command printed 400 lines successfully.')
    expect(seenFallbackArgs!.model).toBe('test-model')
    expect(seenFallbackArgs!.maxOutputTokens).toBe(1024)
    expect(seenFallbackArgs!.userPrompt).toContain('Tool: Bash')
    expect(seenFallbackArgs!.systemPrompt).toContain('compress tool outputs')
  })

  test('empty summary text with failing fallback returns undefined', async () => {
    sideQueryMock.mockResolvedValueOnce({ content: [{ type: 'text', text: '   ' }] })
    const params = baseParams()
    params.fallbackQuery = async () => ''
    expect(await maybeSummarizeToolResultBlock(params)).toBeUndefined()
  })

  test('persistence error returns undefined without calling sideQuery', async () => {
    persistMock.mockRejectedValueOnce(new Error('EACCES'))
    expect(await maybeSummarizeToolResultBlock(baseParams())).toBeUndefined()
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })

  test('aborted parent returns undefined without persistence', async () => {
    const params = baseParams()
    params.parentAbortController.abort(new Error('user-cancel'))
    expect(await maybeSummarizeToolResultBlock(params)).toBeUndefined()
    expect(persistMock).toHaveBeenCalledTimes(0)
  })

  test('subagent results skip entirely', async () => {
    const params = baseParams()
    params.isSubagent = true
    expect(await maybeSummarizeToolResultBlock(params)).toBeUndefined()
    expect(persistMock).toHaveBeenCalledTimes(0)
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })

  test('short output skips entirely', async () => {
    const params = baseParams()
    params.toolResultBlock = {
      ...params.toolResultBlock,
      content: 'short',
    }
    expect(await maybeSummarizeToolResultBlock(params)).toBeUndefined()
    expect(persistMock).toHaveBeenCalledTimes(0)
  })

  test('ignored tool skips entirely', async () => {
    const params = baseParams()
    params.toolName = 'Task'
    expect(await maybeSummarizeToolResultBlock(params)).toBeUndefined()
    expect(persistMock).toHaveBeenCalledTimes(0)
  })

  test('non-finite maxResultSizeChars (Read) skips entirely', async () => {
    const params = baseParams()
    params.toolName = 'Read'
    params.maxResultSizeChars = Number.POSITIVE_INFINITY
    expect(await maybeSummarizeToolResultBlock(params)).toBeUndefined()
    expect(persistMock).toHaveBeenCalledTimes(0)
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })

  test('no-blowup guard: summary larger than raw output keeps raw', async () => {
    // Observed in the wild: a verbose summary of a small-but-over-threshold
    // output. The sideQuery succeeds, but the replacement must not enter
    // context if it isn't smaller. Content must be eligible under the mocked
    // config (lineThreshold 300): 311 one-char lines = 620 chars raw, and a
    // 600-char summary + ~180-char wrapper (~780) loses to it.
    const verboseSummary = 'x'.repeat(600)
    sideQueryMock.mockResolvedValueOnce({
      content: [{ type: 'text', text: verboseSummary }],
    })
    const params = baseParams()
    params.toolResultBlock = {
      ...params.toolResultBlock,
      content: 'x\n'.repeat(310),
    }
    const result = await maybeSummarizeToolResultBlock(params)
    expect(result).toBeUndefined()
    // The call happened (not skipped by floor or eligibility) — the guard
    // discarded the replacement.
    expect(persistMock).toHaveBeenCalledTimes(1)
  })
})
