import { describe, expect, test } from 'bun:test'
import {
  buildSummarizerUserMessage,
  TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT,
} from './prompt.js'

describe('TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT', () => {
  test('states the self-sufficiency contract', () => {
    expect(TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT).toContain('ONLY your summary')
  })

  test('requires verbatim preservation of exact values', () => {
    expect(TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT).toContain('verbatim')
    expect(TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT).toContain('Never invent')
  })
})

describe('buildSummarizerUserMessage', () => {
  test('formats tool name, input, and measured output', () => {
    const message = buildSummarizerUserMessage({
      toolName: 'Bash',
      toolInput: { command: 'ls -la' },
      output: 'file1\nfile2',
      lineCount: 2,
      estimatedTokens: 42,
      truncated: false,
    })

    expect(message).toBe(
      'Tool: Bash\n' +
        'Input: {"command":"ls -la"}\n' +
        'Output (2 lines, ~42 tokens):\n' +
        'file1\nfile2',
    )
  })

  test('marks truncated output', () => {
    const message = buildSummarizerUserMessage({
      toolName: 'Grep',
      toolInput: { pattern: 'TODO' },
      output: 'match',
      lineCount: 1,
      estimatedTokens: 1,
      truncated: true,
    })

    expect(message).toContain('Output (1 lines, ~1 tokens):')
    expect(message).toContain('[output truncated for this summary]')
  })

  test('truncates long input JSON to 500 chars', () => {
    const message = buildSummarizerUserMessage({
      toolName: 'Bash',
      toolInput: { command: 'x'.repeat(2000) },
      output: 'ok',
      lineCount: 1,
      estimatedTokens: 1,
      truncated: false,
    })

    const inputLine = message.split('\n')[1]
    const inputJson = inputLine!.slice('Input: '.length)
    expect(inputJson.length).toBeLessThanOrEqual(500)
    expect(inputJson).toEndWith('...')
  })

  test('handles unserializable input', () => {
    const circular: Record<string, unknown> = {}
    circular.self = circular
    const message = buildSummarizerUserMessage({
      toolName: 'Bash',
      toolInput: circular,
      output: 'ok',
      lineCount: 1,
      estimatedTokens: 1,
      truncated: false,
    })

    expect(message).toContain('[unserializable input]')
  })
})
