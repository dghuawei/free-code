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
import { writeFile } from 'node:fs/promises'
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

/**
 * Raw outputs below this size are never sent to the summarizer: the
 * replacement wrapper (tags + saved-to path) alone costs ~200 chars, so no
 * summary can shrink them. Must stay consistent with buildSummarizedContent.
 */
export const MIN_SUMMARIZABLE_RAW_CHARS = 300

export type MaybeSummarizeToolResultParams = {
  /** The mapped tool_result block whose content may be summarized */
  toolResultBlock: ToolResultBlockParam
  toolName: string
  /**
   * The tool's declared maxResultSizeChars. Non-finite (e.g. Read) marks
   * tools whose verbatim output the model must see — see eligibility check.
   */
  maxResultSizeChars: number
  /** The tool's parsed input, given to the summarizer for context */
  toolInput: unknown
  /** Turn-level controller; user cancellation aborts the summary call */
  parentAbortController: AbortController
  /** Subagent sidechains keep raw outputs — summarization is main-thread only */
  isSubagent: boolean
  /**
   * Overrides the attempt-2 transport (main-loop query path). Tests inject a
   * stub; production leaves it undefined for the default lazy implementation.
   */
  fallbackQuery?: SummarizerFallbackQuery
}

/**
 * Attempt 2 transport: runs the summarization through the main query path
 * (queryModelWithoutStreaming). Injected for tests; the default uses dynamic
 * imports so this module adds no static coupling to the api/claude chain —
 * that chain transitively imports modules that tests mock, and a static
 * import would break their partial mock exports.
 */
export type SummarizerFallbackQuery = (args: {
  userPrompt: string
  systemPrompt: string
  model: string
  maxOutputTokens: number
  signal: AbortSignal
}) => Promise<string>

async function defaultFallbackQuery(args: {
  userPrompt: string
  systemPrompt: string
  model: string
  maxOutputTokens: number
  signal: AbortSignal
}): Promise<string> {
  const [{ queryModelWithoutStreaming }, { getEmptyToolPermissionContext }, { createUserMessage }, { asSystemPrompt }] =
    await Promise.all([
      import('../../services/api/claude.js'),
      import('../../Tool.js'),
      import('../../utils/messages.js'),
      import('../../utils/systemPromptType.js'),
    ])
  const response = await queryModelWithoutStreaming({
    messages: [createUserMessage({ content: args.userPrompt })],
    systemPrompt: asSystemPrompt([args.systemPrompt]),
    thinkingConfig: { type: 'disabled' },
    tools: [],
    signal: args.signal,
    options: {
      getToolPermissionContext: async () => getEmptyToolPermissionContext(),
      model: args.model,
      toolChoice: undefined,
      isNonInteractiveSession: false,
      hasAppendSystemPrompt: false,
      agents: [],
      querySource: 'tool_result_summarization',
      mcpTools: [],
      maxOutputTokensOverride: args.maxOutputTokens,
      skipCacheWrite: true,
    },
  })
  if (response.isApiErrorMessage) return ''
  return (
    (response.message?.content ?? [])
      .filter(
        (block): block is { type: 'text'; text: string } =>
          block.type === 'text',
      )
      .map(block => block.text)
      .join('')
      .trim() ?? ''
  )
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
  maxResultSizeChars: number,
): SummarizationEligibility | null {
  if (!config.enabled || isSubagent) return null
  if (config.ignoredTools.has(toolName)) return null
  if (isToolResultContentEmpty(content)) return null

  // Tools declaring Infinity (Read) need verbatim output — the model re-reads
  // or edits files from this content, and a summary forces a wasteful re-read
  // loop. Same exclusion the budget system applies (query.ts skipToolNames,
  // built from !Number.isFinite(t.maxResultSizeChars)); also matches the
  // persistence path's circularity rationale (toolResultStorage.ts).
  if (!Number.isFinite(maxResultSizeChars)) return null

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

  // Structural floor: the wrapper alone (tags + saved-to line with a real
  // session path) costs ~200 chars, so no summary of output this small can
  // beat the no-blowup guard downstream. Observed live: a 291-char output
  // was persisted and side-queried, only for the 318-char summary to be
  // discarded by the guard. Skipping here avoids the wasted API call and
  // the orphaned persisted file.
  if (text.length < MIN_SUMMARIZABLE_RAW_CHARS) return null

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
  maxResultSizeChars,
  toolInput,
  parentAbortController,
  isSubagent,
  fallbackQuery,
}: MaybeSummarizeToolResultParams): Promise<ToolResultBlockParam | undefined> {
  const config = getToolOutputSummarizationConfig()
  const eligibility = getSummarizationEligibility(
    toolResultBlock.content,
    toolName,
    config,
    isSubagent,
    maxResultSizeChars,
  )
  if (!eligibility) return undefined

  if (parentAbortController.signal.aborted) return undefined
  const { text, lineCount, estimatedTokens } = eligibility

  // Timeout flag is hoisted above the try so the catch can distinguish a
  // timeout abort (SDK rewrites it to a generic "Request was aborted") from
  // other failures.
  let timedOut = false
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
    const timeout = setTimeout(() => {
      timedOut = true
      summarizerAbort.abort(
        new Error('tool result summarization timed out'),
      )
    }, config.timeoutMs)

    const truncated = text.length > config.maxInputChars
    const userPrompt = buildSummarizerUserMessage({
      toolName,
      toolInput,
      output: truncated ? text.slice(0, config.maxInputChars) : text,
      lineCount,
      estimatedTokens,
      truncated,
    })
    const systemPrompt = config.noThinkPromptSuffix
      ? `${TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT}\n\n${config.noThinkPromptSuffix}`
      : TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT

    let summaryText = ''
    // Attempt 1: sideQuery (direct Anthropic protocol). Fast path on
    // first-party providers; on some OpenAI-compatible gateways this
    // endpoint misbehaves (200-with-empty-text, or hangs when streamed).
    try {
      const response = await sideQuery({
        model: config.model,
        system: systemPrompt,
        skipSystemPromptPrefix: true,
        messages: [
          {
            role: 'user',
            content: userPrompt,
          },
        ],
        max_tokens: config.maxOutputTokens,
        maxRetries: 1,
        // Thinking models (qwen, deepseek-r1, …) burn the entire token
        // budget on reasoning and return ZERO text blocks — observed live:
        // 200 OK with empty content, silently skipping every summarization.
        // Compaction output must be plain text.
        thinking: false,
        // Some OpenAI-compatible gateways only populate content on streamed
        // responses; config.streaming opts this call into the stream path.
        stream: config.streaming,
        signal: summarizerAbort.signal,
        querySource: 'tool_result_summarization',
      })
      summaryText = response.content
        .filter(block => block.type === 'text')
        .map(block => (block.type === 'text' ? block.text : ''))
        .join('')
        .trim()
      if (!summaryText) {
        // Make this failure mode visible: a 200-with-no-text response is the
        // signature of a reasoning model (thinking not disabled server-side)
        // or a gateway that drops text blocks. The raw-response dump shows
        // exactly what came back; the fallback below then retries via the
        // main transport.
        const responseDump = JSON.stringify(response)
          .slice(0, 600)
        logForDebugging(
          `Summarizer returned no text for ${toolName} via sideQuery. ` +
            `Request: model=${config.model} stream=${config.streaming} ` +
            `max_tokens=${config.maxOutputTokens} thinking=disabled ` +
            `suffix=${JSON.stringify(config.noThinkPromptSuffix)}. ` +
            `Response: ${responseDump}`,
          { level: 'warn' },
        )
      }
    } catch (error) {
      if (timedOut) {
        logForDebugging(
          `Summarizer sideQuery timed out after ${config.timeoutMs}ms for ${toolName} ` +
            `(model=${config.model}, stream=${config.streaming})`,
          { level: 'warn' },
        )
      } else {
        logError(toError(error))
      }
    } finally {
      clearTimeout(timeout)
    }

    // Attempt 2 (fallback): the main query transport. Forks that wrap the
    // main loop in an OpenAI-compatible transport (where the gateway's
    // native-Anthropic endpoint is broken) still get every tool result
    // summarized — this is the exact path every conversation turn uses,
    // so it is the most reliable route to the model. Fresh timeout budget.
    if (!summaryText && !parentAbortController.signal.aborted) {
      const fallbackAbort = createChildAbortController(parentAbortController)
      let fallbackTimedOut = false
      const fallbackTimeout = setTimeout(() => {
        fallbackTimedOut = true
        fallbackAbort.abort(
          new Error('tool result summarization fallback timed out'),
        )
      }, config.timeoutMs)
      try {
        summaryText = await (fallbackQuery ?? defaultFallbackQuery)({
          userPrompt,
          systemPrompt,
          model: config.model,
          maxOutputTokens: config.maxOutputTokens,
          signal: fallbackAbort.signal,
        })
        if (summaryText) {
          logForDebugging(
            `Summarized ${toolName} via main-transport fallback after sideQuery returned no text`,
            { level: 'info' },
          )
        }
      } catch (error) {
        if (fallbackTimedOut) {
          logForDebugging(
            `Summarizer main-transport fallback timed out after ${config.timeoutMs}ms for ${toolName}`,
            { level: 'warn' },
          )
        } else {
          logError(toError(error))
        }
      } finally {
        clearTimeout(fallbackTimeout)
      }
    }

    if (!summaryText) return undefined

    const replacement = buildSummarizedContent({
      summaryText,
      persisted,
      lineCount,
      originalSize: text.length,
    })

    // No-blowup guard: the wrapper (tags + saved-to path) has a fixed cost of
    // a couple hundred chars, so near-threshold outputs can end up BIGGER as a
    // "summary" (observed: 140-char seq output → 340-char summary). If the
    // replacement doesn't shrink the context, keep the raw output.
    if (replacement.length >= text.length) return undefined

    const summaryCopyPath = await saveSummaryCopy(
      persisted.filepath,
      summaryText,
    )

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
      `Summarized ${toolName} output: ${text.length} chars / ${lineCount} lines -> ${replacement.length} chars (raw saved to ${persisted.filepath}${summaryCopyPath ? `, summary copy: ${summaryCopyPath}` : ''})`,
      { level: 'info' },
    )

    return { ...toolResultBlock, content: replacement }
  } catch (error) {
    // Fail open: log for observability, keep raw output. Timeouts get a
    // distinct message — the SDK rewrites the abort into a generic
    // "Request was aborted", which is indistinguishable from user
    // cancellation in the raw error alone.
    if (timedOut) {
      logForDebugging(
        `Summarizer call timed out after ${config.timeoutMs}ms for ${toolName} ` +
          `— keeping raw output (model=${config.model}, stream=${config.streaming})`,
        { level: 'warn' },
      )
    } else {
      logError(toError(error))
    }
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

/**
 * Debug aid: write the exact summary text that entered the model context
 * next to the persisted raw output as "<name>_sum.<ext>" (e.g.
 * toolu_1_sum.txt), so raw vs summarized can be diffed offline. Fail-open —
 * a write failure never blocks summarization. Returns the path written, or
 * undefined on failure.
 */
async function saveSummaryCopy(
  rawFilepath: string,
  summaryText: string,
): Promise<string | undefined> {
  const dot = rawFilepath.lastIndexOf('.')
  const summaryPath =
    dot > 0
      ? `${rawFilepath.slice(0, dot)}_sum${rawFilepath.slice(dot)}`
      : `${rawFilepath}_sum`
  try {
    await writeFile(summaryPath, summaryText, 'utf-8')
    return summaryPath
  } catch (error) {
    logForDebugging(
      `Failed to save summarized output copy to ${summaryPath}: ${toError(error).message}`,
      { level: 'warn' },
    )
    return undefined
  }
}
