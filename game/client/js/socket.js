// Minimal reconnecting WebSocket. Behavior adapted from
// web/src/lib/resilientWebSocket.ts (unmodified source file): exponential
// backoff with jitter, generation guard against obsolete sockets, and a
// stale-message watchdog (the server heartbeats every 2 s, so silence longer
// than staleAfterMs on an OPEN socket means a half-dead connection).

export class GameSocket {
  constructor(url, { onMessage, onStatus, staleAfterMs = 6000 }) {
    this.url = url
    this.onMessage = onMessage
    this.onStatus = onStatus
    this.generation = 0
    this.attempt = 0
    this.socket = null
    this.closed = false
    this.lastMessageAt = performance.now()
    this.watchdog = setInterval(() => {
      if (this.closed || !this.socket) return
      if (
        this.socket.readyState === WebSocket.OPEN &&
        performance.now() - this.lastMessageAt > staleAfterMs
      ) {
        console.warn('[game] socket stale — forcing reconnect')
        this.socket.close()
      }
    }, 2000)
    this.connect()
  }

  connect() {
    if (this.closed) return
    const generation = ++this.generation
    const socket = new WebSocket(this.url)
    this.socket = socket
    socket.onopen = () => {
      if (generation !== this.generation) return
      this.attempt = 0
      this.lastMessageAt = performance.now()
      this.onStatus(true)
    }
    socket.onmessage = (event) => {
      if (generation !== this.generation) return
      this.lastMessageAt = performance.now()
      let payload
      try {
        payload = JSON.parse(event.data)
      } catch {
        return
      }
      try {
        this.onMessage(payload)
      } catch (error) {
        console.error('[game] message handling failed', payload?.type, error)
      }
    }
    const retry = () => {
      if (generation !== this.generation || this.closed) return
      this.onStatus(false)
      this.attempt += 1
      const delay = Math.min(8000, 250 * 2 ** this.attempt) * (0.7 + Math.random() * 0.6)
      setTimeout(() => {
        if (generation === this.generation && !this.closed) this.connect()
      }, delay)
    }
    socket.onclose = retry
    socket.onerror = () => socket.close()
  }

  send(payload) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(payload))
      return true
    }
    return false
  }

  close() {
    this.closed = true
    this.generation += 1
    clearInterval(this.watchdog)
    if (this.socket) this.socket.close()
  }
}
