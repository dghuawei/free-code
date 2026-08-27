/**
 * Summarization of API-injected tool results that arrive inside ASSISTANT
 * messages instead of the client tool pipeline.
 *
 * Two producer shapes:
 *  1. Compat layers (e.g. GLM's Anthropic-compatible endpoint) that surface
 *     their native web search as plain `tool_result` blocks inside assistant
 *     messages — no tool_use exists, the client never executed anything.
 *     These blobs are echoed back to the API on every subsequent turn and
 *     observed at multi-KB sizes.
 *  2. First-party server-tool result blocks (web_search_tool_result,
 *     mcp_tool_result, code_execution_tool_result, ...) carrying large
 *     content arrays.
 *
 * Both are summarized ONCE at creation time — inside query.ts's stream loop,
 * before the assistant message is yielded to the UI or pushed into the
 * per-turn message array. Because the mutated message is novel content that
 * has never been part of any prior API request, there is no prompt-cache
 * byte-mismatch window (unlike post-hoc history rewriting, which needs the
 * budget system's frozen-replacement machinery). Resume-safe by the same
 * argument: the transcript stores the already-summarized message.
 *
 * Fail-open contract mirrors the client-tool summarizer: on any failure the
 * original block is kept unchanged.
 */

import { logEvent } from '../analytics/index.js'
import { sanitizeToolNameForAnalytics } from '../analytics/metadata.js'
import type { ToolResultBlockParam } from '@anthropic-ai/sdk/resources/index.mjs'
import {
  getSummarizationEligibility,
  maybeSummarizeToolResultBlock,
} from './summarizeToolResult.js'
import { TOOL_OUTPUT_SUMMARY_TAG } from './tags.js'
import { getToolOutputSummarizationConfig } from './config.js'

/**
 * Server-tool result block types whose content arrays may be summarized.
 * `server_tool_use` / `mcp_tool_use` are the request halves and carry no
 * result payload — excluded. `container_upload` and `compaction` are not
 * results either.
 */
export const SERVER_TOOL_RESULT_BLOCK_TYPES = [
  'web_search_tool_result',
  'mcp_tool_result',
  'code_execution_tool_result',
  'web_fetch_tool_result',
  'bash_code_execution_tool_result',
  'text_editor_code_execution_tool_result',
  'tool_search_tool_result',
] as const

export type ServerToolResultBlockType =
  (typeof SERVER_TOOL_RESULT_BLOCK_TYPES)[number]

/** Short name used for ignoredTools matching and the summarizer prompt. */
export function serverToolNameForBlockType(
  blockType: ServerToolResultBlockType,
): string {
  return blockType.replace(/_tool_result$/, '')
}

/** Tool name reported for compat-injected plain tool_result blocks. */
export const COMPAT_SERVER_TOOL_NAME = 'server_tool_result'

type ContentBlock = { type: string } & Record<string, unknown>

type CollectResult = {
  /** Indexes into message.message.content of blocks to process */
  blockIndexes: number[]
  /** Per-index description of what produced the block */
  kinds: Map<number, 'compat-tool-result' | ServerToolResultBlockType>
}

/**
 * Structural assistant-message shape. Kept local (runtime-guarded) so this
 * module doesn't depend on the types/message.js barrel.
 */
type AssistantMessageLike = {
  type: 'assistant'
  message: { content: ContentBlock[] }
}

function isAssistantMessageLike(m: unknown): m is AssistantMessageLike {
  return (
    typeof m === 'object' &&
    m !== null &&
    (m as { type?: unknown }).type === 'assistant' &&
    typeof (m as { message?: unknown }).message === 'object' &&
    Array.isArray(
      ((m as { message?: { content?: unknown } }).message ?? {}).content,
    )
  )
}

/**
 * Find provider-injected result blocks in an assistant message.
 *
 * Plain `tool_result` blocks are provider-injected by construction: the
 * client pipeline emits tool results as USER messages (toolExecution.ts
 * addToolResult), never as assistant content.
 *
 * Server-tool family blocks additionally respect the
 * `summarizeServerToolResults` config flag.
 */
export function collectServerToolResultBlocks(
  message: unknown,
  summarizeServerToolResults: boolean,
): CollectResult {
  const result: CollectResult = { blockIndexes: [], kinds: new Map() }
  if (!isAssistantMessageLike(message)) return result
  for (let i = 0; i < message.message.content.length; i++) {
    const block = message.message.content[i]!
    if (block.type === 'tool_result') {
      // Compat-injected. Content must be a string for a faithful swap;
      // block-content arrays with images etc. are left alone.
      if (typeof block.content === 'string') {
        result.blockIndexes.push(i)
        result.kinds.set(i, 'compat-tool-result')
      }
      continue
    }
    if (
      summarizeServerToolResults &&
      (SERVER_TOOL_RESULT_BLOCK_TYPES as readonly string[]).includes(
        block.type,
      )
    ) {
      result.blockIndexes.push(i)
      result.kinds.set(
        i,
        block.type as ServerToolResultBlockType,
      )
    }
  }
  return result
}

/**
 * Extract the text a server-tool block contributes to the model context.
 * web_search_result items carry an opaque `encrypted_content` blob the
 * summarizer can't use — strip it and keep the human-readable fields
 * (title, url). Other shapes are stringified whole.
 */
function extractServerBlockText(block: ContentBlock): string {
  const content = block.content
  if (Array.isArray(content)) {
    const cleaned = content.map(item => {
      if (
        typeof item === 'object' &&
        item !== null &&
        'encrypted_content' in item
      ) {
        const { encrypted_content: _drop, ...rest } = item as Record<
          string,
          unknown
        >
        return rest
      }
      return item
    })
    return JSON.stringify(cleaned, null, 1)
  }
  return typeof content === 'string' ? content : JSON.stringify(content)
}

/** True when a block was already summarized by a previous pass of this module. */
function isBlockSummarized(block: ContentBlock): boolean {
  if (typeof block.content === 'string') {
    return block.content.startsWith(TOOL_OUTPUT_SUMMARY_TAG)
  }
  // Server-tool family: the block itself is emptied and the summary lives in
  // the sibling text block that follows it.
  if (Array.isArray(block.content) && block.content.length === 0) return true
  return false
}

export type MaybeSummarizeServerToolResultsParams = {
  /** The assistant message just received from the API stream */
  message: unknown
  /** Turn-level controller; cancellation aborts in-flight summarizer calls */
  parentAbortController: AbortController
  /** Subagent sidechains skip — consistent with the client-tool summarizer */
  isSubagent: boolean
}

/**
 * Summarize provider-injected result blocks in a freshly-streamed assistant
 * message. Returns a NEW message object when any block was replaced, or the
 * input reference unchanged otherwise (zero-cost for ordinary messages).
 * Never throws; failures leave blocks untouched (fail open).
 */
export async function maybeSummarizeServerToolResults({
  message,
  parentAbortController,
  isSubagent,
}: MaybeSummarizeServerToolResultsParams): Promise<unknown> {
  const config = getToolOutputSummarizationConfig()
  if (!config.enabled || isSubagent) return message
  if (parentAbortController.signal.aborted) return message

  const collected = collectServerToolResultBlocks(
    message,
    config.summarizeServerToolResults,
  )
  if (collected.blockIndexes.length === 0) return message
  if (!isAssistantMessageLike(message)) return message

  // Block-level screen before any I/O: extract text once and delegate the
  // floor/threshold/ignore decision to getSummarizationEligibility — the
  // single authority shared with the client-tool path — so both features
  // can never disagree on what qualifies.
  type Pending = {
    index: number
    toolUseId: string
    toolName: string
    text: string
    isCompatToolResult: boolean
  }
  const pending: Pending[] = []
  for (const index of collected.blockIndexes) {
    const block = message.message.content[index]!
    if (isBlockSummarized(block)) continue
    const kind = collected.kinds.get(index)!
    const isCompatToolResult = kind === 'compat-tool-result'
    const toolUseId =
      typeof block.tool_use_id === 'string' && block.tool_use_id.length > 0
        ? block.tool_use_id
        : undefined
    // Without a stable tool_use_id the persisted raw-output reference can't
    // be keyed reliably (persistToolResult dedupes by id) — skip rather than
    // risk a summary pointing at another block's file.
    if (toolUseId === undefined) continue
    const toolName = isCompatToolResult
      ? COMPAT_SERVER_TOOL_NAME
      : serverToolNameForBlockType(kind as ServerToolResultBlockType)
    const text = isCompatToolResult
      ? (block.content as string)
      : extractServerBlockText(block)
    if (
      getSummarizationEligibility(text, toolName, config, false, 100_000) ===
      null
    ) {
      continue
    }
    pending.push({
      index,
      toolUseId,
      toolName,
      text,
      isCompatToolResult,
    })
  }
  if (pending.length === 0) return message

  // Summarize blocks concurrently; each call is independently fail-open.
  const summarized = await Promise.all(
    pending.map(async p => {
      const synthetic: ToolResultBlockParam = {
        type: 'tool_result',
        tool_use_id: p.toolUseId,
        content: p.text,
      }
      const replacement = await maybeSummarizeToolResultBlock({
        toolResultBlock: synthetic,
        toolName: p.toolName,
        maxResultSizeChars: 100_000,
        toolInput: undefined,
        parentAbortController,
        isSubagent: false,
      })
      return replacement?.content as string | undefined
    }),
  )

  let contentChanged = false
  const newContent: ContentBlock[] = [...message.message.content]
  // Apply replacements right-to-left: inserting a sibling text block after a
  // server-tool block shifts the indexes of every LATER block, so descending
  // order keeps pending[].index values valid throughout.
  for (let i = pending.length - 1; i >= 0; i--) {
    const p = pending[i]!
    const replacement = summarized[i]
    if (typeof replacement !== 'string' || replacement.length === 0) continue
    const block = newContent[p.index]!

    if (p.isCompatToolResult) {
      newContent[p.index] = { ...block, content: replacement }
      contentChanged = true
    } else {
      // Server-tool family: empty the result array (block identity and
      // tool_use_id stay — any use/result pairing validation stays
      // satisfied) and carry the summary in a sibling text block.
      newContent[p.index] = { ...block, content: [] }
      newContent.splice(p.index + 1, 0, {
        type: 'text',
        text: replacement,
      })
      contentChanged = true
    }
    logEvent('tengu_server_tool_result_summarized', {
      toolName: sanitizeToolNameForAnalytics(p.toolName),
      originalSizeBytes: p.text.length,
      summarizedSizeBytes: replacement.length,
    })
  }
  if (!contentChanged) return message

  return {
    ...message,
    message: { ...message.message, content: newContent },
  }
}
