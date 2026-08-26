/**
 * QuerySource identifies which subsystem fired an API call. Used for
 * analytics attribution (tengu_api_success COGS joining) and behavior
 * switches (cache TTL, persistence routing, stop hooks).
 *
 * Template patterns cover dynamically-composed sources:
 * - `agent:${string}` — Task tool sidechains (builtin, custom, forked)
 * - `repl_main_thread:${string}` — main thread variants (e.g. output styles)
 */
export type QuerySource =
  | 'repl_main_thread'
  | `repl_main_thread:${string}`
  | 'sdk'
  | 'hook_agent'
  | 'hook_prompt'
  | `agent:${string}`
  | 'agent_creation'
  | 'agent_summary'
  | 'verification_agent'
  | 'compact'
  | 'auto_dream'
  | 'auto_mode'
  | 'auto_mode_critique'
  | 'away_summary'
  | 'bash_extract_prefix'
  | 'chrome_mcp'
  | 'extract_memories'
  | 'feedback'
  | 'generate_session_title'
  | 'insights'
  | 'magic_docs'
  | 'marble_origami'
  | 'mcp_datetime_parse'
  | 'memdir_relevance'
  | 'model_validation'
  | 'permission_explainer'
  | 'prompt_suggestion'
  | 'rename_generate_name'
  | 'session_memory'
  | 'session_search'
  | 'side_question'
  | 'skill_improvement_apply'
  | 'speculation'
  | 'teleport_generate_title'
  | 'tool_result_summarization'
  | 'tool_use_summary_generation'
  | 'web_fetch_apply'
  | 'web_search_tool'
