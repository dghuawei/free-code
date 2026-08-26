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

export type ToolOutputSummarizationConfig = {
  enabled: boolean
  tokenThreshold: number
  lineThreshold: number
  model: string
  ignoredTools: ReadonlySet<string>
  maxInputChars: number
  timeoutMs: number
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
  }
}
