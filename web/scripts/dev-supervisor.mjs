import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const viteEntrypoint = fileURLToPath(
  new URL('../node_modules/vite/bin/vite.js', import.meta.url),
)

if (!existsSync(viteEntrypoint)) {
  console.error('[observatory] Dependencies are missing. Run `npm install` first.')
  process.exit(1)
}

let child = null
let stopping = false

function stop(signal) {
  stopping = true
  if (child && !child.killed) child.kill(signal)
}

process.once('SIGINT', () => stop('SIGINT'))
process.once('SIGTERM', () => stop('SIGTERM'))

while (!stopping) {
  child = spawn(
    process.execPath,
    [viteEntrypoint, '--host', '127.0.0.1'],
    { stdio: 'inherit' },
  )
  const result = await new Promise((resolve) => {
    child.once('exit', (code, signal) => resolve({ code, signal }))
  })
  child = null
  if (stopping) break
  console.error(
    `[observatory] Visualization exited (${result.signal ?? result.code}); restarting in 1 second…`,
  )
  await new Promise((resolve) => setTimeout(resolve, 1000))
}
