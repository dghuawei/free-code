/**
 * Live demo view for tool-output summarization.
 *
 * When enabled (toolOutputSummarization.liveViewFile), a single file per
 * session — <session>/tool-results/live-view.txt — is fully rewritten at each
 * stage of the cycle: raw tool output on arrival (written by toolExecution
 * for EVERY tool result), then the summary once it enters model context
 * (written by the summarizer). Watching the file (tail -f / Get-Content
 * -Wait) shows the raw-to-summary replacement the feature performs.
 *
 * Every write is a complete truncate-and-rewrite (writeFile default flag),
 * so concurrent tool results intentionally clobber each other — the file
 * always reflects exactly one stage of one result, which is the point.
 * Fail-open: write errors are logged and never affect tool results.
 */
import { writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { logForDebugging } from '../../utils/debug.js'
import { toError } from '../../utils/errors.js'
import { getToolResultsDir } from '../../utils/toolResultStorage.js'
import { BYTES_PER_TOKEN } from '../../constants/toolLimits.js'
import { getToolOutputSummarizationConfig } from './config.js'

export const LIVE_VIEW_FILENAME = 'live-view.txt'

export type LiveToolOutputViewStage = 'raw' | 'summary'

/** Flatten a tool_result block's content to plain text for the view file. */
function contentToText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content
      .map(block => {
        if (
          typeof block === 'object' &&
          block !== null &&
          'type' in block &&
          (block as { type: string }).type === 'text' &&
          'text' in block
        ) {
          return String((block as { text: unknown }).text)
        }
        return ''
      })
      .filter(text => text !== '')
      .join('\n')
  }
  try {
    return JSON.stringify(content, null, 2) ?? ''
  } catch {
    return String(content)
  }
}

/**
 * Rewrite the live-view file with one stage of one tool result. No-op unless
 * the setting is enabled. Never throws.
 */
export async function writeLiveToolOutputView({
  stage,
  toolName,
  content,
  originalTokens,
}: {
  stage: LiveToolOutputViewStage
  toolName: string
  content: unknown
  /** Estimated original size in tokens; required for the summary stage */
  originalTokens?: number
}): Promise<string | undefined> {
  if (!getToolOutputSummarizationConfig().liveViewFile) return undefined
  const text = contentToText(content)
  const tokens = Math.ceil(text.length / BYTES_PER_TOKEN)
  const header =
    stage === 'summary' && originalTokens !== undefined
      ? `=== ${new Date().toISOString()} · ${toolName} · SUMMARY (~${originalTokens} -> ~${tokens} tokens) ===`
      : `=== ${new Date().toISOString()} · ${toolName} · RAW (${text.length} chars, ~${tokens} tokens) ===`
  const filepath = join(getToolResultsDir(), LIVE_VIEW_FILENAME)
  try {
    await writeFile(filepath, `${header}\n${text}\n`, 'utf-8')
    return filepath
  } catch (error) {
    logForDebugging(
      `Failed to write live tool output view ${filepath}: ${toError(error).message}`,
      { level: 'warn' },
    )
    return undefined
  }
}
