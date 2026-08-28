/**
 * Parser tests for the summarization badge fields. These exercise the
 * structural extraction from the tool_result block content (the badge's
 * only data channel — no UI rendering here).
 */
import { describe, expect, test } from 'bun:test'
import {
  getSummarizedOutputPath,
  getSummarizedTokenDelta,
} from './badgeParsing.js'

function messageWithBlockContent(content: unknown) {
  return {
    message: {
      content: [
        { type: 'tool_result', tool_use_id: 'toolu_1', content },
      ],
    },
  } as Parameters<typeof getSummarizedTokenDelta>[0]
}

const SUMMARIZED =
  '<tool-output-summary>\n' +
  'Digest of the output.\n' +
  '</tool-output-summary>\n' +
  'Full output (400 lines, 3.4KB) saved to: /session/tool-results/toolu_1.txt\n' +
  'Estimated tokens: 873 -> 67'

describe('getSummarizedTokenDelta', () => {
  test('extracts "<original> -> <after>" from a summarized block', () => {
    expect(
      getSummarizedTokenDelta(messageWithBlockContent(SUMMARIZED), 'toolu_1'),
    ).toBe('873 -> 67')
  })

  test('null for raw (unsummarized) and missing blocks', () => {
    expect(
      getSummarizedTokenDelta(messageWithBlockContent('plain raw output'), 'toolu_1'),
    ).toBeNull()
    expect(getSummarizedTokenDelta(messageWithBlockContent(SUMMARIZED), 'toolu_other')).toBeNull()
  })

  test('null when the wrapper predates the token line', () => {
    const legacy =
      '<tool-output-summary>\nx\n</tool-output-summary>\n' +
      'Full output (10 lines, 1KB) saved to: /x'
    expect(getSummarizedTokenDelta(messageWithBlockContent(legacy), 'toolu_1')).toBeNull()
  })

  test('path parser still resolves alongside the token delta', () => {
    expect(
      getSummarizedOutputPath(messageWithBlockContent(SUMMARIZED), 'toolu_1'),
    ).toBe('/session/tool-results/toolu_1.txt')
  })
})
