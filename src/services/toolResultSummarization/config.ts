// Leaf config module for tool-output summarization — minimal imports so
// toolExecution.ts can read the resolved config without pulling in the
// summarizer/sideQuery chain (avoids cycles, keeps cold path light).

import { getSmallFastModel } from '../../utils/model/model.js'
import { getInitialSettings } from '../../utils/settings/settings.js'
import { isEnvTruthy } from '../../utils/envUtils.js'

export const DEFAULT_TOKEN_THRESHOLD = 2500
export const DEFAULT_LINE_THRESHOLD = 300
export const DEFAULT_MAX_INPUT_CHARS = 200_000
export const DEFAULT_TIMEOUT_MS = 30_000
export const DEFAULT_MAX_OUTPUT_TOKENS = 1024

export type ToolOutputSummarizationConfig = {
  enabled: boolean
  tokenThreshold: number
  lineThreshold: number
  model: string
  ignoredTools: ReadonlySet<string>
  maxInputChars: number
  timeoutMs: number
  summarizeServerToolResults: boolean
  /** Appended to the summarizer system prompt; Qwen3 soft switch is "/no_think" */
  noThinkPromptSuffix: string
  /** max_tokens for the summarizer call; raise for reasoning models */
  maxOutputTokens: number
  /** Stream the summarizer call; for gateways that only populate content on streams */
  streaming: boolean
  /** Rewrite <session>/tool-results/live-view.txt per stage (raw, then summary) */
  liveViewFile?: boolean
}

/**
 * Resolved tool-output summarization config. Unset fields fall back to
 * defaults; the env kill-switch DISABLE_TOOL_OUTPUT_SUMMARIZATION wins over
 * everything (escape hatch for debugging context issues).
 */
export function getToolOutputSummarizationConfig(): ToolOutputSummarizationConfig {
  const setting = getInitialSettings().toolOutputSummarization ?? {}
  return {
    enabled:
      !isEnvTruthy(process.env.DISABLE_TOOL_OUTPUT_SUMMARIZATION) &&
      setting.enabled !== false,
    tokenThreshold: setting.tokenThreshold ?? DEFAULT_TOKEN_THRESHOLD,
    lineThreshold: setting.lineThreshold ?? DEFAULT_LINE_THRESHOLD,
    model: setting.model ?? getSmallFastModel(),
    ignoredTools: new Set(setting.ignoredTools ?? []),
    maxInputChars: setting.maxInputChars ?? DEFAULT_MAX_INPUT_CHARS,
    timeoutMs: setting.timeoutMs ?? DEFAULT_TIMEOUT_MS,
    summarizeServerToolResults: setting.summarizeServerToolResults === true,
    noThinkPromptSuffix: setting.noThinkPromptSuffix ?? '',
    maxOutputTokens: setting.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS,
    streaming: setting.streaming === true,
    liveViewFile: setting.liveViewFile === true,
  }
}
