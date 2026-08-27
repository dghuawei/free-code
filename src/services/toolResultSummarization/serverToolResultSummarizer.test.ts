import { describe, expect, test } from 'bun:test'
import {
  collectServerToolResultBlocks,
  COMPAT_SERVER_TOOL_NAME,
  serverToolNameForBlockType,
} from './serverToolResultSummarizer.js'

function assistantMessage(content: unknown[]) {
  return {
    type: 'assistant',
    message: { role: 'assistant', content },
  }
}

describe('collectServerToolResultBlocks', () => {
  test('detects plain tool_result blocks in assistant messages', () => {
    const msg = assistantMessage([
      { type: 'text', text: 'hi' },
      { type: 'tool_result', tool_use_id: 'call_1', content: 'blob' },
    ])
    const result = collectServerToolResultBlocks(msg, false)
    expect(result.blockIndexes).toEqual([1])
    expect(result.kinds.get(1)).toBe('compat-tool-result')
  })

  test('non-string tool_result content is skipped', () => {
    const msg = assistantMessage([
      {
        type: 'tool_result',
        tool_use_id: 'call_1',
        content: [{ type: 'image', source: {} }],
      },
    ])
    expect(collectServerToolResultBlocks(msg, false).blockIndexes).toEqual([])
  })

  test('server-tool family blocks require the config flag', () => {
    const block = {
      type: 'web_search_tool_result',
      tool_use_id: 'srvrtu_1',
      content: [{ type: 'web_search_result', url: 'https://x', title: 'x' }],
    }
    expect(
      collectServerToolResultBlocks(assistantMessage([block]), false)
        .blockIndexes,
    ).toEqual([])
    const enabled = collectServerToolResultBlocks(
      assistantMessage([block]),
      true,
    )
    expect(enabled.blockIndexes).toEqual([0])
    expect(enabled.kinds.get(0)).toBe('web_search_tool_result')
  })

  test('server_tool_use blocks are never collected', () => {
    const msg = assistantMessage([
      { type: 'server_tool_use', id: 'srvrtu_1', name: 'web_search', input: {} },
    ])
    expect(collectServerToolResultBlocks(msg, true).blockIndexes).toEqual([])
  })

  test('non-assistant messages yield nothing', () => {
    const userMsg = {
      type: 'user',
      message: {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 't', content: 'x' }],
      },
    }
    expect(collectServerToolResultBlocks(userMsg, true).blockIndexes).toEqual(
      [],
    )
    expect(collectServerToolResultBlocks(null, true).blockIndexes).toEqual([])
  })
})

describe('serverToolNameForBlockType', () => {
  test('strips the _tool_result suffix', () => {
    expect(serverToolNameForBlockType('web_search_tool_result')).toBe(
      'web_search',
    )
    expect(serverToolNameForBlockType('mcp_tool_result')).toBe('mcp')
    expect(serverToolNameForBlockType('code_execution_tool_result')).toBe(
      'code_execution',
    )
  })

  test('compat blocks report the fixed name', () => {
    expect(COMPAT_SERVER_TOOL_NAME).toBe('server_tool_result')
  })
})
