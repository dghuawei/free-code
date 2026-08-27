/**
 * Integration tests for maybeSummarizeToolResultBlock with the sideQuery,
 * persistence, and config boundaries mocked. Verifies the fail-open contract:
 * every failure mode returns undefined so raw output enters context.
 *
 * mock.module is global in bun — afterAll restores to avoid leaking into
 * sibling test files.
 */
import { afterAll, beforeEach, describe, expect, mock, test } from 'bun:test'

const FIXED_CONFIG = {
  enabled: true,
  tokenThreshold: 2500,
  lineThreshold: 300,
  model: 'test-model',
  ignoredTools: new Set<string>(['Task']),
  maxInputChars: 200_000,
  timeoutMs: 30_000,
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
    sideQueryMock.mockResolvedValueOnce({
      content: [{ type: 'text', text: 'Listed 400 files; all under src/.' }],
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
        'Full output (400 lines, 3.4KB) saved to: /session/tool-results/toolu_1.txt',
    )
    expect(persistMock).toHaveBeenCalledTimes(1)
  })

  test('sideQuery failure returns undefined (fail open)', async () => {
    sideQueryMock.mockRejectedValueOnce(new Error('429 rate limited'))
    const params = baseParams()

    expect(await maybeSummarizeToolResultBlock(params)).toBeUndefined()
  })

  test('empty summary text returns undefined', async () => {
    sideQueryMock.mockResolvedValueOnce({ content: [{ type: 'text', text: '   ' }] })
    expect(await maybeSummarizeToolResultBlock(baseParams())).toBeUndefined()
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
