/**
 * Background summarization of long tool outputs.
 *
 * When a tool output exceeds the configured line or token threshold, the raw
 * output is persisted to the session tool-results directory and a side-query
 * model call produces a summary that replaces the raw output in the model's
 * context, keeping conversation history lean.
 *
 * Fail-open contract: on any failure (sideQuery error, timeout, persistence
 * error, abort) the caller receives `undefined` and the raw output enters
 * context unchanged — the feature must never lose or block a tool result.
 */

import type { ToolResultBlockParam } from '@anthropic-ai/sdk/resources/index.mjs'
import { BYTES_PER_TOKEN } from '../../constants/toolLimits.js'
import { logEvent } from '../analytics/index.js'
import { sanitizeToolNameForAnalytics } from '../analytics/metadata.js'
import { createChildAbortController } from '../../utils/abortController.js'
import { logForDebugging } from '../../utils/debug.js'
import { toError } from '../../utils/errors.js'
import { formatFileSize } from '../../utils/format.js'
import { logError } from '../../utils/log.js'
import { sideQuery } from '../../utils/sideQuery.js'
import {
  hasImageBlock,
  isContentAlreadyCompacted,
  isToolResultContentEmpty,
  persistToolResult,
  type PersistedToolResult,
  type PersistToolResultError,
} from '../../utils/toolResultStorage.js'
import { getToolOutputSummarizationConfig } from './config.js'
import {
  buildSummarizerUserMessage,
  TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT,
} from './prompt.js'
import {
  TOOL_OUTPUT_SUMMARY_CLOSING_TAG,
  TOOL_OUTPUT_SUMMARY_TAG,
} from './tags.js'

export type MaybeSummarizeToolResultParams = {
  /** The mapped tool_result block whose content may be summarized */
  toolResultBlock: ToolResultBlockParam
  toolName: string
  /** The tool's parsed input, given to the summarizer for context */
  toolInput: unknown
  /** Turn-level controller; user cancellation aborts the summary call */
  parentAbortController: AbortController
  /** Subagent sidechains keep raw outputs — summarization is main-thread only */
  isSubagent: boolean
}

type SummarizationEligibility = {
  text: string
  lineCount: number
  estimatedTokens: number
}

/**
 * Decide whether content qualifies for summarization and extract the text it
 * would operate on. Pure — no I/O, no config reads. Returns null when the
 * content must not be summarized.
 */
export function getSummarizationEligibility(
  content: ToolResultBlockParam['content'],
  toolName: string,
  config: ReturnType<typeof getToolOutputSummarizationConfig>,
  isSubagent: boolean,
): SummarizationEligibility | null {
  if (!config.enabled || isSubagent) return null
  if (config.ignoredTools.has(toolName)) return null
  if (isToolResultContentEmpty(content)) return null

  // Extract the text the summarizer would see. Only string content or arrays
  // of purely text blocks qualify — anything else (images, mixed blocks)
  // cannot be faithfully represented as text.
  let text: string
  if (typeof content === 'string') {
    text = content
  } else if (Array.isArray(content) && content.every(b => b.type === 'text')) {
    text = content.map(b => (b.type === 'text' ? b.text : '')).join('\n')
  } else {
    return null
  }

  // Image arrays were excluded above; keep the explicit guard for clarity at
  // the call boundary in case extraction rules change.
  if (hasImageBlock(content)) return null

  // Already compacted by the 50k persistence path or a prior summarization —
  // never re-process (double-summarizing a preview loses information).
  if (isContentAlreadyCompacted(content)) return null

  const lineCount = text.split('\n').length
  const estimatedTokens = Math.ceil(text.length / BYTES_PER_TOKEN)
  if (lineCount <= config.lineThreshold && estimatedTokens <= config.tokenThreshold) {
    return null
  }

  return { text, lineCount, estimatedTokens }
}

/**
 * Maybe summarize a long tool result. Returns a replacement block whose
 * content is a summary referencing the persisted raw output, or `undefined`
 * to leave the original untouched. Never throws.
 */
export async function maybeSummarizeToolResultBlock({
  toolResultBlock,
  toolName,
  toolInput,
  parentAbortController,
  isSubagent,
}: MaybeSummarizeToolResultParams): Promise<ToolResultBlockParam | undefined> {
  const config = getToolOutputSummarizationConfig()
  const eligibility = getSummarizationEligibility(
    toolResultBlock.content,
    toolName,
    config,
    isSubagent,
  )
  if (!eligibility) return undefined

  if (parentAbortController.signal.aborted) return undefined
  const { text, lineCount, estimatedTokens } = eligibility

  try {
    // Persist the raw output first so the summary can reference it for
    // recovery. Idempotent ('wx' + EEXIST reuse), so the fallback persistence
    // path re-writing the same content after a failed summary is a no-op.
    const persisted = await persistToolResult(
      toolResultBlock.content,
      toolResultBlock.tool_use_id,
    )
    if (isPersistError(persisted)) return undefined

    const summarizerAbort = createChildAbortController(parentAbortController)
    // Bounded wait: on timeout the raw output enters context instead. The
    // summarizer's own retries share this budget.
    const timeout = setTimeout(
      () =>
        summarizerAbort.abort(
          new Error('tool result summarization timed out'),
        ),
      config.timeoutMs,
    )

    let summaryText: string
    try {
      const truncated = text.length > config.maxInputChars
      const response = await sideQuery({
        model: config.model,
        system: TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT,
        skipSystemPromptPrefix: true,
        messages: [
          {
            role: 'user',
            content: buildSummarizerUserMessage({
              toolName,
              toolInput,
              output: truncated ? text.slice(0, config.maxInputChars) : text,
              lineCount,
              estimatedTokens,
              truncated,
            }),
          },
        ],
        max_tokens: 1024,
        maxRetries: 1,
        signal: summarizerAbort.signal,
        querySource: 'tool_result_summarization',
      })
      summaryText = response.content
        .filter(block => block.type === 'text')
        .map(block => (block.type === 'text' ? block.text : ''))
        .join('')
        .trim()
    } finally {
      clearTimeout(timeout)
    }

    if (!summaryText) return undefined

    const replacement = buildSummarizedContent({
      summaryText,
      persisted,
      lineCount,
      originalSize: text.length,
    })

    logEvent('tengu_tool_result_summarized', {
      toolName: sanitizeToolNameForAnalytics(toolName),
      originalSizeBytes: text.length,
      summarizedSizeBytes: replacement.length,
      estimatedOriginalTokens: estimatedTokens,
      estimatedSummarizedTokens: Math.ceil(replacement.length / BYTES_PER_TOKEN),
    })
    // Visible with -d/--debug: confirms the model context got the summary,
    // not the raw output (the terminal UI still renders the raw output from
    // toolUseResult by design).
    logForDebugging(
      `Summarized ${toolName} output: ${text.length} chars / ${lineCount} lines -> ${replacement.length} chars (saved to ${persisted.filepath})`,
      { level: 'info' },
    )

    return { ...toolResultBlock, content: replacement }
  } catch (error) {
    // Fail open: log for observability, keep raw output.
    logError(toError(error))
    return undefined
  }
}

export function buildSummarizedContent({
  summaryText,
  persisted,
  lineCount,
  originalSize,
}: {
  summaryText: string
  persisted: PersistedToolResult
  lineCount: number
  originalSize: number
}): string {
  return (
    `${TOOL_OUTPUT_SUMMARY_TAG}\n` +
    `${summaryText}\n` +
    `${TOOL_OUTPUT_SUMMARY_CLOSING_TAG}\n` +
    `Full output (${lineCount} lines, ${formatFileSize(originalSize)}) saved to: ${persisted.filepath}`
  )
}

function isPersistError(
  result: PersistedToolResult | PersistToolResultError,
): result is PersistToolResultError {
  return 'error' in result
}
