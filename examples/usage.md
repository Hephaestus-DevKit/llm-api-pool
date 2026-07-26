# Usage Examples

## OpenAI Compatible (most tools)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello from the pool!"}],
    "stream": false
  }'
```

Python:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="any")
resp = client.chat.completions.create(model="auto", messages=[{"role":"user","content":"hi"}])
print(resp.choices[0].message.content)
```

For Cursor / Continue.dev / Aider: set base_url to http://localhost:8080/v1 , api_key any.

## Anthropic Compatible (Claude Code, etc.)

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello from Anthropic format!"}]
  }'
```

With anthropic SDK (set base_url):

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8080/v1", api_key="any")
msg = client.messages.create(model="auto", max_tokens=1024, messages=[{"role":"user", "content":"hi"}])
print(msg.content[0].text)
```

For Claude Code CLI: set ANTHROPIC_BASE_URL=http://localhost:8080/v1

## Tool Calling

Tools work in either dialect and are translated to whatever the chosen channel speaks, so
an Anthropic-style request can be served by an OpenAI channel and the reply still comes back
in Anthropic form.

```bash
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "name": "get_weather",
      "description": "Current weather for a city",
      "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }]
  }'
```

The reply carries a `tool_use` block. Send the result back as a `tool_result` block in the
next turn, exactly as you would against the real API.

Web-session and Gemini channels cannot carry tool calls. When a request includes tools, the
router scores tool-capable channels higher so the call is not silently answered without them.

## Monitoring

`GET /admin/status` returns per-channel health, quota, in-flight count, latency, success
rate, cooldown state, and whether the channel supports tools. It needs the admin token:

```bash
curl http://localhost:8080/admin/status -H "X-Admin-Token: <token>"
```

The dashboard at `/` shows the same data visually.

## Adding Web Sessions

Pasted cookies are far more reliable than password login, which only works for simple
accounts without 2FA, captcha, or SSO:

```json
{
  "type": "web_codex",
  "name": "my-codex",
  "cookies": {"__Secure-next-auth.session-token": "..."}
}
```

Password mode is also accepted and drives a headless login:

```json
{
  "type": "web_codex",
  "name": "my-codex",
  "email": "you@example.com",
  "password": "yourpass"
}
```

If the login does not produce a usable session cookie the request fails instead of creating a
channel that would break on its first real call.

**Tip:** for high volume, prefer official API keys. Use `web_*` for extra quota or when you
have no key.

## Model Selection and Failover

`"auto"` lets the router pick. A concrete name biases toward matching channels: `claude-*`
only reaches Claude channels, `gpt-*` only OpenAI-compatible ones. If nothing matches, the
pool returns `503` naming the model rather than answering from the wrong provider — set
`CROSS_PROVIDER_FALLBACK=1` to change that.

Per-channel `aliases` map your own names onto a backend model:

```json
{
  "type": "official_claude",
  "api_key": "sk-ant-...",
  "aliases": {"fast": "claude-haiku-4-5-20251001", "smart": "claude-opus-5"}
}
```

Requests that fail upstream are automatically retried on the next-best channel, so one dead
account does not surface as a failed request.
