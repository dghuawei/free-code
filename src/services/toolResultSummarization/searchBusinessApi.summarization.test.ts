/**
 * Summarization contract tests for SearchBusinessApiTool's output shapes.
 *
 * Why this file exists: Search_business_api is a CLIENT tool (standard
 * tool_result in a user message), so `summarizeServerToolResults` never
 * applies to it — only the master flag + thresholds do. Its mapped output
 * (mapToolResultToToolResultBlockParam) drops inputSchema/outputSchema and
 * emits one `N. **bapi__<domain>__<name>**: <desc>` line per result, which
 * is normally compact (under thresholds → correctly raw). These tests pin:
 *   1. compact guidance stays raw (no blowup),
 *   2. oversized guidance (large topk / long descriptions) summarizes,
 *   3. registered tool identifiers survive summarization verbatim — the
 *      agent calls bapi__ tools by exact name from the summary alone.
 *
 * The tool itself can't be imported here (its framework ecosystem —
 * BusinessApiTool/, services/businessApi/, apiIndex/ — is not in this
 * repo), so the mapped block shapes are replicated from
 * SearchBusinessApiTool.ts mapToolResultToToolResultBlockParam.
 */
import { afterAll, beforeEach, describe, expect, mock, test } from 'bun:test'

const FIXED_CONFIG = {
  enabled: true,
  // Default thresholds — the contract that governs this tool in production.
  tokenThreshold: 2_500,
  lineThreshold: 300,
  model: 'test-model',
  ignoredTools: new Set<string>([]),
  maxInputChars: 200_000,
  timeoutMs: 5_000,
  summarizeServerToolResults: false,
    noThinkPromptSuffix: "",
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

const { maybeSummarizeToolResultBlock } = await import(
  './summarizeToolResult.js'
)

/** Mirrors SearchBusinessApiTool.mapToolResultToToolResultBlockParam. */
function buildMappedSearchResult(
  apis: Array<{ domain: string; apiName: string; apiDesc: string }>,
): string {
  const formatted = apis
    .map(
      (api, idx) =>
        `${idx + 1}. **bapi__${api.domain}__${api.apiName}**: ${api.apiDesc}`,
    )
    .join('\n')
  return (
    `Found ${apis.length} matching business APIs. All have been automatically ` +
    `registered as callable tools:\n\n${formatted}`
  )
}

function searchParams(content: string) {
  return {
    toolResultBlock: {
      type: 'tool_result' as const,
      tool_use_id: 'toolu_bapi_search',
      content,
    },
    toolName: 'Search_business_api',
    maxResultSizeChars: 50_000,
    toolInput: { query: 'check order status' },
    parentAbortController: new AbortController(),
    isSubagent: false,
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

describe('SearchBusinessApiTool output summarization', () => {
  test('typical topk=10 result stays raw — compact by design, under default thresholds', async () => {
    const content = buildMappedSearchResult(
      Array.from({ length: 10 }, (_, i) => ({
        domain: 'order',
        apiName: `getOrderDetail${i}`,
        apiDesc: `Fetch order detail ${i} including status, items, and shipping info.`,
      })),
    )
    expect(content.length).toBeLessThan(2_000) // sanity: genuinely compact

    const result = await maybeSummarizeToolResultBlock(searchParams(content))
    expect(result).toBeUndefined()
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })

  test('oversized result (large topk, long descriptions) is summarized with tool names preserved', async () => {
    // 60 APIs × ~400-char descriptions ≈ 25KB → over the 2500-token threshold.
    const apis = Array.from({ length: 60 }, (_, i) => ({
      domain: `domain${i % 5}`,
      apiName: `queryBusinessRecord${i}`,
      apiDesc:
        `Query business record ${i}. `.repeat(8) +
        'Returns record identifier, creation timestamp, owner, status, line items, amounts, currency, and audit trail.',
    }))
    const content = buildMappedSearchResult(apis)
    expect(Math.ceil(content.length / 4)).toBeGreaterThan(2_500)

    // A realistic summary that keeps the callable identifiers.
    const summaryText =
      'Success. 60 APIs registered as callable tools, grouped by domain:\n' +
      '- bapi__domain0__queryBusinessRecord0/5/10/…/55: query business records (id, timestamp, owner, status, items, amounts, currency, audit trail)\n' +
      '- bapi__domain1__queryBusinessRecord1/6/…: same shape, domain1 tenant\n' +
      'Full names for all 60 in the saved file.'
    sideQueryMock.mockResolvedValueOnce({
      content: [{ type: 'text', text: summaryText }],
    })

    const result = await maybeSummarizeToolResultBlock(searchParams(content))
    expect(result).toBeDefined()
    const summarized = result!.content as string
    expect(summarized).toContain('<tool-output-summary>')
    expect(summarized).toContain('bapi__domain0__queryBusinessRecord0')
    expect(summarized).toContain('saved to: /session/tool-results/toolu_bapi_search.txt')
    expect(summarized.length).toBeLessThan(content.length)
  })

  test('full-JSON variant (schemas mapped into content, e.g. a framework build) summarizes', async () => {
    // Some builds may expose the raw structured output; verify the same path
    // handles JSON.stringify'd {code,message,data:[…inputSchema…]} payloads.
    const payload = JSON.stringify({
      code: 200,
      message: 'success',
      data: Array.from({ length: 40 }, (_, i) => ({
        domain: 'order',
        apiName: `getOrder${i}`,
        apiDesc: 'Fetch order with full detail.',
        inputSchema: {
          type: 'object',
          properties: Object.fromEntries(
            Array.from({ length: 15 }, (_, p) => [
              `param${p}`,
              { type: 'string', description: 'x'.repeat(60) },
            ]),
          ),
          required: ['param0'],
        },
        outputSchema: { type: 'object' },
      })),
    })
    expect(Math.ceil(payload.length / 4)).toBeGreaterThan(2_500)

    sideQueryMock.mockResolvedValueOnce({
      content: [
        { type: 'text', text: 'Success. 40 order APIs registered; param0 required on each.' },
      ],
    })
    const result = await maybeSummarizeToolResultBlock(searchParams(payload))
    expect(result).toBeDefined()
    expect((result!.content as string)).toContain('<tool-output-summary>')
  })

  test('ignoredTools excludes it by exact name Search_business_api', async () => {
    FIXED_CONFIG.ignoredTools = new Set(['Search_business_api'])
    try {
      const content = buildMappedSearchResult(
        Array.from({ length: 60 }, (_, i) => ({
          domain: 'd',
          apiName: `a${i}`,
          apiDesc: 'd'.repeat(300),
        })),
      )
      const result = await maybeSummarizeToolResultBlock(searchParams(content))
      expect(result).toBeUndefined()
      expect(sideQueryMock).toHaveBeenCalledTimes(0)
    } finally {
      FIXED_CONFIG.ignoredTools = new Set()
    }
  })

  test('error path maps to a short message that stays raw', async () => {
    // mapToolResultToToolResultBlockParam failure branch.
    const content = 'Search failed or no matching results: ToolBank search failed: Internal Server Error'
    const result = await maybeSummarizeToolResultBlock(searchParams(content))
    expect(result).toBeUndefined()
    expect(sideQueryMock).toHaveBeenCalledTimes(0)
  })
})
