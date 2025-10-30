Vercel-ready LINE Bot (Python/FastAPI)

1) What this is
- Minimal FastAPI app that handles LINE webhook on /callback and calls OpenAI Chat Completions.
- Uses Postgres (recommended) if DATABASE_URL is provided; otherwise falls back to stateless memory mode (not for production).

2) Files
- vercel.json       : Vercel routing & runtime config (Python 3.11).
- requirements.txt  : Python dependencies.
- api/index.py      : ASGI app exported as `app` (required by Vercel).

3) Environment variables (configure in Vercel Project Settings → Environment Variables)
- LINE_CHANNEL_ACCESS_TOKEN : Your LINE Messaging API channel access token
- LINE_CHANNEL_SECRET       : Your LINE Messaging API channel secret
- OPENAI_API_KEY            : Your OpenAI API key
- OPENAI_MODEL              : (optional) default 'gpt-4o-mini'
- DATABASE_URL              : (recommended) e.g. postgresql+psycopg2://user:pass@host:5432/dbname

4) Database notes
- Serverless functions have ephemeral file systems; avoid SQLite files. Use a cloud Postgres (Neon/Supabase/Vercel Postgres).
- SQLAlchemy is configured with NullPool to avoid cross-invocation connection reuse problems.

5) Deploy
- vercel login
- vercel --prod

6) LINE webhook
- Set your LINE webhook URL to: https://<your-vercel-domain>/callback
- Add a simple GET https://<your-vercel-domain>/health for uptime checks.

7) Local run
- pip install -r requirements.txt
- uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload

