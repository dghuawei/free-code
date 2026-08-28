/**
 * Structural parsers for the summarization badge. The badge's only data
 * channel is the tool_result block content string itself (the transcript
 * may deserialize without types), so extraction lives here — free of UI
 * imports — where both the component and tests can reach it.
 */
import type { NormalizedUserMessage } from '../../types/message.js'
import { TOOL_OUTPUT_SUMMARY_TAG } from './tags.js'

function findToolResultBlockContent(
  message: NormalizedUserMessage,
  toolUseID: string,
): unknown {
  const content = (message as { message?: { content?: unknown } }).message
    ?.content
  if (!Array.isArray(content)) return undefined
  const block = content.find(
    (b: { type?: string; tool_use_id?: string }) =>
      b?.type === 'tool_result' && b?.tool_use_id === toolUseID,
  )
  return block?.content
}

/**
 * Path of the persisted raw output for a summarized block ('' when
 * summarized but no path found), else null. The rendered output above stays
 * raw (toolUseResult), so the badge is the only on-screen signal that the
 * model actually received a summary. Structural access + runtime guards
 * because the transcript path may deserialize without types.
 */
export function getSummarizedOutputPath(
  message: NormalizedUserMessage,
  toolUseID: string,
): string | null {
  const blockContent = findToolResultBlockContent(message, toolUseID)
  if (
    typeof blockContent !== 'string' ||
    !blockContent.startsWith(TOOL_OUTPUT_SUMMARY_TAG)
  ) {
    return null
  }
  const match = blockContent.match(/saved to: (.+)$/m)
  return match?.[1] ?? ''
}

/**
 * "<original_tokens> -> <tokens_after_summarization>" from the summary
 * wrapper's "Estimated tokens:" line, else null. Both are char-based
 * estimates (tool results have no billing usage of their own).
 */
export function getSummarizedTokenDelta(
  message: NormalizedUserMessage,
  toolUseID: string,
): string | null {
  const blockContent = findToolResultBlockContent(message, toolUseID)
  if (
    typeof blockContent !== 'string' ||
    !blockContent.startsWith(TOOL_OUTPUT_SUMMARY_TAG)
  ) {
    return null
  }
  const match = blockContent.match(/Estimated tokens: (\d+) -> (\d+)/)
  return match ? `${match[1]} -> ${match[2]}` : null
}
