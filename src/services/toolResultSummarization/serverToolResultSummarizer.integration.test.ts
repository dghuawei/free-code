/**
 * Integration tests for maybeSummarizeServerToolResults with the sideQuery
 * boundary mocked. Verifies the once-at-creation mutation of provider-
 * injected assistant-message blocks and the fail-open contract.
 */
import { afterAll, beforeEach, describe, expect, mock, test } from 'bun:test'

const FIXED_CONFIG = {
  enabled: true,
  tokenThreshold: 200,
  lineThreshold: 20,
  model: 'test-model',
  ignoredTools: new Set<string>([]),
  maxInputChars: 200_000,
  timeoutMs: 5_000,
  summarizeServerToolResults: true,
}

mock.module('./config.js', () => ({
  DEFAULT_TOKEN_THRESHOLD: 2500,
  DEFAULT_LINE_THRESHOLD: 300,
  DEFAULT_MAX_INPUT_CHARS: 200_000,
  DEFAULT_TIMEOUT_MS: 30_000,
  getToolOutputSummarizationConfig: () => FIXED_CONFIG,
}))

const sideQueryMock = mock(
  async (
    _opts?: unknown,
  ): Promise<{ content: Array<{ type: string; text: string }> }> => {
    throw new Error('sideQuery mock not configured for this test')
  },
)

mock.module('../../utils/sideQuery.js', () => ({
  sideQuery: sideQueryMock,
}))

const persistMock = mock(async (content: unknown, id: string) => ({
  filepath: `/session/tool-results/${id}.txt`,
  originalSize: typeof content === 'string' ? content.length : 0,
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
}))

const { maybeSummarizeServerToolResults } = await import(
  './serverToolResultSummarizer.js'
)

const BIG = 'x'.repeat(2_000) // > token threshold (200) and floor (300)

function assistantWithCompatBlob() {
  return {
    type: 'assistant' as const,
    message: {
      role: 'assistant' as const,
      id: 'msg_1',
      content: [
        { type: 'text', text: 'Searching the web…' },
        { type: 'tool_result', tool_use_id: 'call_glm_1', content: BIG },
      ],
    },
  }
}

function assistantWithServerBlock() {
  return {
    type: 'assistant' as const,
    message: {
      role: 'assistant' as const,
      id: 'msg_2',
      content: [
        { type: 'server_tool_use', id: 'srvrtu_1', name: 'web_search', input: { query: 'q' } },
        {
          type: 'web_search_tool_result',
          tool_use_id: 'srvrtu_1',
          content: [
            { type: 'web_search_result', url: 'https://a', title: 'A', encrypted_content: 'AAAA' },
            { type: 'web_search_result', url: 'https://b', title: 'B', encrypted_content: 'BBBB' },
            ...Array.from({ length: 30 }, (_, i) => ({
              type: 'web_search_result',
              url: `https://x${i}`,
              title: `X${i}`,
              encrypted_content: 'E'.repeat(100),
            })),
          ],
        },
        { type: 'text', text: 'Done.' },
      ],
    },
  }
}

beforeEach(() => {
  persistMock.mockClear()
  sideQueryMock.mockClear()
  sideQueryMock.mockImplementation(async () => {
    throw new Error('sideQuery mock not configured for this test')
  })
})

afterAll(() => {
  mock.restore()
})

describe('maybeSummarizeServerToolResults', () => {
  test('compat tool_result blob: content replaced with tagged summary', async () => {
    sideQueryMock.mockResolvedValueOnce({
      content: [{ type: 'text', text: 'Search returned 3 results about topic X.' }],
    })
    const input = assistantWithCompatBlob()
    const result = (await maybeSummarizeServerToolResults({
      message: input,
      parentAbortController: new AbortController(),
      isSubagent: false,
    })) as ReturnType<typeof assistantWithCompatBlob>

    const block = result.message.content[1] as { type: string; content: string }
    expect(block.type).toBe('tool_result')
    expect(block.content).toContain('<tool-output-summary>')
    expect(block.content).toContain('Search returned 3 results about topic X.')
    expect(block.content).toContain('saved to: /session/tool-results/call_glm_1.txt')
    // Block count unchanged — compat path swaps content in place.
    expect(result.message.content).toHaveLength(2)
    // Immutable style: the input message object keeps the raw blob.
    expect(result).not.toBe(input)
    expect(
      (input.message.content[1] as { content: string }).content,
    ).toHaveLength(2_000)
  })

  test('server-tool block: emptied array + sibling text block carries summary', async () => {
    sideQueryMock.mockResolvedValueOnce({
      content: [{ type: 'text', text: '32 search results; top: A, B.' }],
    })
    const input = assistantWithServerBlock()
    const result = (await maybeSummarizeServerToolResults({
      message: input,
      parentAbortController: new AbortController(),
      isSubagent: false,
    })) as ReturnType<typeof assistantWithServerBlock>

    const blocks = result.message.content
    const resultBlock = blocks[1] as { type: string; tool_use_id: string; content: unknown[] }
    expect(resultBlock.type).toBe('web_search_tool_result')
    expect(resultBlock.content).toEqual([])
    expect(blocks[2]!.type).toBe('text')
    expect((blocks[2] as { text: string }).text).toContain('<tool-output-summary>')
    // One sibling inserted.
    expect(blocks).toHaveLength(4)
    // Original input untouched (immutable style).
    expect(
      (input.message.content[1] as { content: unknown[] }).content.length,
    ).toBe(32)
  })

  test('two server blocks: sibling insertions do not shift the second replacement', async () => {
    sideQueryMock.mockResolvedValue({
      content: [{ type: 'text', text: 'summary text here' }],
    })
    // Visible (post-strip) text must clear the floor + token threshold —
    // encrypted_content is stripped before screening.
    const results = (tag: string) =>
      Array.from({ length: 5 }, (_, i) => ({
        type: 'web_search_result',
        url: `https://${tag}-${i}.example.com/some/path`,
        title: 'T'.repeat(120),
        encrypted_content: 'E'.repeat(2_000),
      }))
    const input = {
      type: 'assistant' as const,
      message: {
        role: 'assistant' as const,
        id: 'msg_3',
        content: [
          { type: 'web_search_tool_result', tool_use_id: 'srvrtu_a', content: results('a') },
          { type: 'text', text: 'middle' },
          { type: 'web_search_tool_result', tool_use_id: 'srvrtu_b', content: results('b') },
        ],
      },
    }
    const result = (await maybeSummarizeServerToolResults({
      message: input,
      parentAbortController: new AbortController(),
      isSubagent: false,
    })) as typeof input

    const blocks = result.message.content
    // Both emptied, each followed by its sibling summary.
    expect((blocks[0] as { content: unknown[] }).content).toEqual([])
    expect(blocks[1]!.type).toBe('text')
    expect((blocks[3] as { content: unknown[] }).content).toEqual([])
    expect(blocks[4]!.type).toBe('text')
    expect(blocks).toHaveLength(5)
  })

  test('encrypted_content is stripped from the summarizer input', async () => {
    let seenPrompt = ''
    sideQueryMock.mockImplementationOnce(async (opts: unknown) => {
      seenPrompt = (
        (opts as { messages: Array<{ content: string }> }).messages[0]!
          .content
      )
      return { content: [{ type: 'text', text: 's' }] }
    })
    await maybeSummarizeServerToolResults({
      message: assistantWithServerBlock(),
      parentAbortController: new AbortController(),
      isSubagent: false,
    })
    expect(seenPrompt).not.toContain('encrypted_content')
    expect(seenPrompt).toContain('https://a')
  })

  test('sideQuery failure keeps the message unchanged (fail open)', async () => {
    sideQueryMock.mockRejectedValueOnce(new Error('429'))
    const input = assistantWithCompatBlob()
    const result = await maybeSummarizeServerToolResults({
      message: input,
      parentAbortController: new AbortController(),
      isSubagent: false,
    })
    expect(result).toBe(input)
  })

  test('already-summarized blocks are skipped (idempotent)', async () => {
    const msg = assistantMessageAlreadySummarized()
    const result = await maybeSummarizeServerToolResults({
      message: msg,
      parentAbortController: new AbortController(),
      isSubagent: false,
    })
    expect(result).toBe(msg)
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })

  test('subagent messages skip entirely', async () => {
    const input = assistantWithCompatBlob()
    const result = await maybeSummarizeServerToolResults({
      message: input,
      parentAbortController: new AbortController(),
      isSubagent: true,
    })
    expect(result).toBe(input)
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })

  test('ordinary assistant messages return the same reference, zero calls', async () => {
    const input = {
      type: 'assistant' as const,
      message: {
        role: 'assistant' as const,
        id: 'msg_4',
        content: [
          { type: 'text', text: 'hello' },
          { type: 'tool_use', id: 't1', name: 'Bash', input: { command: 'ls' } },
        ],
      },
    }
    const result = await maybeSummarizeServerToolResults({
      message: input,
      parentAbortController: new AbortController(),
      isSubagent: false,
    })
    expect(result).toBe(input)
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })

  test('small compat blob (under floor) skipped', async () => {
    const msg = {
      type: 'assistant' as const,
      message: {
        role: 'assistant' as const,
        id: 'msg_5',
        content: [
          { type: 'tool_result', tool_use_id: 'call_small', content: 'tiny blob' },
        ],
      },
    }
    const result = await maybeSummarizeServerToolResults({
      message: msg,
      parentAbortController: new AbortController(),
      isSubagent: false,
    })
    expect(result).toBe(msg)
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })
})

function assistantMessageAlreadySummarized() {
  return {
    type: 'assistant' as const,
    message: {
      role: 'assistant' as const,
      id: 'msg_6',
      content: [
        {
          type: 'tool_result',
          tool_use_id: 'call_done',
          content:
            '<tool-output-summary>\nalready summarized\n</tool-output-summary>\nFull output (10 lines, 1KB) saved to: /x',
        },
        {
          type: 'web_search_tool_result',
          tool_use_id: 'srvrtu_done',
          content: [],
        },
      ],
    },
  }
}
