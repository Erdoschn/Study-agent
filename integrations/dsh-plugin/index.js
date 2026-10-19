import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'study-agent'
export const inject = ['tools']

export function apply(ctx) {
  ctx.tools.register(defineTool({
    name: 'study_agent_health',
    description: 'Check whether the configured Study Agent API is reachable.',
    parameters: {},
    output: {
      schema: {
        type: 'object',
        properties: { status: { type: 'string' } },
        required: ['status'],
        additionalProperties: false,
      },
      render: (_args, value) => [{ type: 'text', text: value.status }],
    },
    async execute(_args, exec) {
      const apiUrl = process.env.STUDY_AGENT_API_URL || 'http://127.0.0.1:8001/v1/chat/completions'
      const url = new URL(apiUrl)
      url.pathname = '/health'
      url.search = ''
      url.hash = ''
      const response = await fetch(url, {
        headers: process.env.STUDY_AGENT_WEB_API_KEY
          ? { Authorization: 'Bearer ' + process.env.STUDY_AGENT_WEB_API_KEY }
          : {},
        redirect: 'error',
        signal: exec.signal,
      })
      if (!response.ok) throw new Error('Study Agent API health check failed: HTTP ' + response.status)
      const value = await response.json()
      return { status: String(value.status || 'unknown') }
    },
  }))
}
