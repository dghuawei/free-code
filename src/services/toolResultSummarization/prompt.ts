// Prompt for background tool-output summarization. The summary replaces the
// raw output in the model's context, so it must be self-sufficient: the agent
// may take actions based on it without ever seeing the raw output (only the
// persisted-file path is recoverable via a re-read).

export const TOOL_OUTPUT_SUMMARY_SYSTEM_PROMPT = `You compress tool outputs for a coding agent. The agent continues its task using ONLY your summary, so it must contain every fact needed to act.

You receive a tool name, its inputs, and the full output.

Rules:
1. State the outcome first: success, failure, partial, or empty result.
2. Preserve verbatim anything the agent must reuse or reference exactly: file paths, line numbers, error messages, stack-trace frames, exit codes, identifiers, URLs, commands, counts, and changed values.
3. Keep structure that aids scanning: terse bullets; preserve section order when it carries meaning (e.g. test file grouping).
4. Drop: decorative output, progress indicators, timestamps, banners, repeated similar lines (report the pattern, a count, and one example), and data the agent cannot act on.
5. If the output reveals a problem, quote the single most diagnostic line.
6. Never invent or guess values. If something is ambiguous, say so in one short clause.

Style: plain text, no preamble ("The tool returned..."), no markdown headers. Target under 150 words unless dense technical detail (errors, paths, key/value facts) requires more; never exceed 300 words.`

const MAX_INPUT_JSON_CHARS = 500

export type BuildSummarizerUserMessageParams = {
  toolName: string
  toolInput: unknown
  output: string
  lineCount: number
  estimatedTokens: number
  truncated: boolean
}

export function buildSummarizerUserMessage({
  toolName,
  toolInput,
  output,
  lineCount,
  estimatedTokens,
  truncated,
}: BuildSummarizerUserMessageParams): string {
  const inputStr = truncateForPrompt(safeJsonStringify(toolInput), MAX_INPUT_JSON_CHARS)
  const truncationNote = truncated ? '\n[output truncated for this summary]' : ''
  return (
    `Tool: ${toolName}\n` +
    `Input: ${inputStr}\n` +
    `Output (${lineCount} lines, ~${estimatedTokens} tokens):${truncationNote}\n` +
    output
  )
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? 'null'
  } catch {
    return '[unserializable input]'
  }
}

function truncateForPrompt(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str
  return str.slice(0, maxLength - 3) + '...'
}
