import { describe, expect, test } from 'bun:test'
import type { ToolResultBlockParam } from '@anthropic-ai/sdk/resources/index.mjs'
import { getToolOutputSummarizationConfig } from './config.js'
import {
  buildSummarizedContent,
  getSummarizationEligibility,
  type MaybeSummarizeToolResultParams,
} from './summarizeToolResult.js'

const config = getToolOutputSummarizationConfigForTests()

function getToolOutputSummarizationConfigForTests() {
  // Fixed config so eligibility tests don't depend on the machine's
  // settings.json or env.
  return {
    enabled: true,
    tokenThreshold: 2500,
    lineThreshold: 300,
    model: 'test-model',
    ignoredTools: new Set(['Task']),
    maxInputChars: 200_000,
    timeoutMs: 30_000,
    summarizeServerToolResults: false,
  }
}

function blockWithContent(content: ToolResultBlockParam['content']): ToolResultBlockParam {
  return {
    type: 'tool_result',
    tool_use_id: 'toolu_test',
    content,
  }
}

describe('getSummarizationEligibility', () => {
  test('long lines trigger via line threshold', () => {
    const text = 'a\n'.repeat(400)
    const eligibility = getSummarizationEligibility(
      text,
      'Bash',
      config,
      false,
      30_000,
    )
    expect(eligibility).not.toBeNull()
    // split('\n') counts the empty string after the trailing newline
    expect(eligibility!.lineCount).toBe(401)
    expect(eligibility!.estimatedTokens).toBe(Math.ceil(text.length / 4))
  })

  test('single huge line triggers via token threshold', () => {
    const text = 'x'.repeat(2500 * 4 + 1)
    const eligibility = getSummarizationEligibility(text, 'Bash', config, false, 30_000)
    expect(eligibility).not.toBeNull()
    expect(eligibility!.lineCount).toBe(1)
  })

  test('short output is not summarized', () => {
    expect(
      getSummarizationEligibility('short output', 'Bash', config, false, 30_000),
    ).toBeNull()
  })

  test('both thresholds under limit is not summarized', () => {
    // 299 lines, ~2.4 tokens — below both.
    const text = Array.from({ length: 299 }, () => 'ok').join('\n')
    expect(getSummarizationEligibility(text, 'Bash', config, false, 30_000)).toBeNull()
  })

  test('either threshold exceeded triggers (lines within, tokens over)', () => {
    // 10 lines but each 2000 chars → ~5000 tokens.
    const text = Array.from({ length: 10 }, () => 'y'.repeat(2000)).join('\n')
    expect(getSummarizationEligibility(text, 'Bash', config, false, 30_000)).not.toBeNull()
  })

  test('below MIN_SUMMARIZABLE_RAW_CHARS returns null even over threshold', () => {
    // Over the line threshold but only 75 chars — the ~200-char wrapper
    // makes any summary a net blowup, so skip before any I/O.
    const lowLineConfig = { ...config, lineThreshold: 20 }
    const text = 'ab\n'.repeat(25)
    expect(text.length).toBeLessThan(300)
    expect(
      getSummarizationEligibility(text, 'Bash', lowLineConfig, false, 30_000),
    ).toBeNull()
  })

  test('disabled config returns null', () => {
    expect(
      getSummarizationEligibility('a\n'.repeat(400), 'Bash', { ...config, enabled: false }, false, 30_000),
    ).toBeNull()
  })

  test('non-finite maxResultSizeChars (Read) returns null — model needs verbatim file content', () => {
    expect(
      getSummarizationEligibility(
        'a\n'.repeat(400),
        'Read',
        config,
        false,
        Number.POSITIVE_INFINITY,
      ),
    ).toBeNull()
  })

  test('finite maxResultSizeChars summarizes as before', () => {
    expect(
      getSummarizationEligibility('a\n'.repeat(400), 'Bash', config, false, 30_000),
    ).not.toBeNull()
  })

  test('subagent results return null', () => {
    expect(
      getSummarizationEligibility('a\n'.repeat(400), 'Bash', config, true, 30_000),
    ).toBeNull()
  })

  test('ignored tool returns null', () => {
    expect(
      getSummarizationEligibility('a\n'.repeat(400), 'Task', config, false, 30_000),
    ).toBeNull()
  })

  test('ignored MCP tool by full name returns null', () => {
    const mcpConfig = {
      ...config,
      ignoredTools: new Set(['mcp__github__list_issues']),
    }
    expect(
      getSummarizationEligibility(
        'a\n'.repeat(400),
        'mcp__github__list_issues',
        mcpConfig,
        false,
        30_000,
      ),
    ).toBeNull()
  })

  test('empty content returns null', () => {
    expect(getSummarizationEligibility('', 'Bash', config, false, 30_000)).toBeNull()
    expect(getSummarizationEligibility('   \n  ', 'Bash', config, false, 30_000)).toBeNull()
    expect(
      getSummarizationEligibility(undefined, 'Bash', config, false, 30_000),
    ).toBeNull()
  })

  test('image block content returns null', () => {
    const content: ToolResultBlockParam['content'] = [
      { type: 'text', text: 'a\n'.repeat(400) },
      {
        type: 'image',
        source: {
          type: 'base64',
          media_type: 'image/png',
          data: 'x',
        },
      },
    ]
    expect(
      getSummarizationEligibility(content, 'Bash', config, false, 30_000),
    ).toBeNull()
  })

  test('already-compacted content returns null', () => {
    expect(
      getSummarizationEligibility(
        '<persisted-output>\nstuff\n</persisted-output>',
        'Bash',
        config,
        false,
        30_000,
      ),
    ).toBeNull()
    expect(
      getSummarizationEligibility(
        '<tool-output-summary>\nstuff\n</tool-output-summary>',
        'Bash',
        config,
        false,
        30_000,
      ),
    ).toBeNull()
  })

  test('all-text block array is eligible and joined', () => {
    const longText = 'a\n'.repeat(400)
    const content = [
      { type: 'text' as const, text: longText },
      { type: 'text' as const, text: 'tail' },
    ]
    const eligibility = getSummarizationEligibility(content, 'Bash', config, false, 30_000)
    expect(eligibility).not.toBeNull()
    expect(eligibility!.text).toBe(`${longText}\ntail`)
  })
})

describe('buildSummarizedContent', () => {
  test('byte-exact golden format', () => {
    const content = buildSummarizedContent({
      summaryText: 'Command succeeded; listed 3 files.',
      persisted: {
        filepath: '/tmp/session/tool-results/toolu_abc.txt',
        originalSize: 12_000,
        isJson: false,
        preview: '...',
        hasMore: true,
      },
      lineCount: 350,
      originalSize: 12_000,
    })

    expect(content).toBe(
      '<tool-output-summary>\n' +
        'Command succeeded; listed 3 files.\n' +
        '</tool-output-summary>\n' +
        'Full output (350 lines, 11.7KB) saved to: /tmp/session/tool-results/toolu_abc.txt',
    )
  })

  test('content starts with the summary tag for compaction detection', () => {
    const content = buildSummarizedContent({
      summaryText: 'summary',
      persisted: {
        filepath: '/x',
        originalSize: 1000,
        isJson: false,
        preview: '',
        hasMore: false,
      },
      lineCount: 10,
      originalSize: 1000,
    })
    expect(content.startsWith('<tool-output-summary>')).toBe(true)
  })
})

// Compile-time presence check: the wired params shape must stay in sync with
// toolExecution.ts's call site.
test('MaybeSummarizeToolResultParams carries the wired call-site fields', () => {
  const params: MaybeSummarizeToolResultParams = {
    toolResultBlock: blockWithContent('ok'),
    toolName: 'Bash',
    maxResultSizeChars: 30_000,
    toolInput: { command: 'ls' },
    parentAbortController: new AbortController(),
    isSubagent: false,
  }
  expect(params.toolName).toBe('Bash')
})
