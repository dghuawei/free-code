import { afterEach, describe, expect, test } from 'bun:test'
import {
  DEFAULT_LINE_THRESHOLD,
  DEFAULT_MAX_INPUT_CHARS,
  DEFAULT_TIMEOUT_MS,
  DEFAULT_TOKEN_THRESHOLD,
  getToolOutputSummarizationConfig,
} from './config.js'

const SAVED_ENV = process.env.DISABLE_TOOL_OUTPUT_SUMMARIZATION

afterEach(() => {
  if (SAVED_ENV === undefined) {
    delete process.env.DISABLE_TOOL_OUTPUT_SUMMARIZATION
  } else {
    process.env.DISABLE_TOOL_OUTPUT_SUMMARIZATION = SAVED_ENV
  }
})

describe('getToolOutputSummarizationConfig', () => {
  test('env kill-switch disables regardless of settings', () => {
    process.env.DISABLE_TOOL_OUTPUT_SUMMARIZATION = '1'
    expect(getToolOutputSummarizationConfig().enabled).toBeFalse()
    process.env.DISABLE_TOOL_OUTPUT_SUMMARIZATION = 'true'
    expect(getToolOutputSummarizationConfig().enabled).toBeFalse()
  })

  test('documented defaults are internally consistent', () => {
    expect(DEFAULT_TOKEN_THRESHOLD).toBe(2500)
    expect(DEFAULT_LINE_THRESHOLD).toBe(300)
    expect(DEFAULT_MAX_INPUT_CHARS).toBe(200_000)
    expect(DEFAULT_TIMEOUT_MS).toBe(30_000)
  })
})
