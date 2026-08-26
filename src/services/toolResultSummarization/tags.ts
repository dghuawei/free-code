// XML tags wrapping summarized tool-output content. Leaf module (no imports)
// so both the summarizer and toolResultStorage's compaction detection share
// one authoritative definition without an import cycle.

export const TOOL_OUTPUT_SUMMARY_TAG = '<tool-output-summary>'
export const TOOL_OUTPUT_SUMMARY_CLOSING_TAG = '</tool-output-summary>'
