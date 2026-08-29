export interface ResilientSocketOptions {
  url: string
  onOpen: () => void
  onMessage: (event: MessageEvent) => void
  onDisconnect: (reason: string) => void
  onRetry?: (delayMs: number, attempt: number) => void
  socketFactory?: (url: string) => WebSocket
  now?: () => number
  random?: () => number
  baseRetryMs?: number
  maxRetryMs?: number
  connectTimeoutMs?: number
  staleTimeoutMs?: number
  watchdogIntervalMs?: number
}

/** A browser WebSocket that survives failed handshakes, silent stalls, and restarts. */
export class ResilientWebSocket {
  private socket: WebSocket | null = null
  private retryTimer: ReturnType<typeof setTimeout> | null = null
  private connectTimer: ReturnType<typeof setTimeout> | null = null
  private watchdogTimer: ReturnType<typeof setInterval> | null = null
  private generation = 0
  private attempt = 0
  private stopped = true
  private lastMessageAt = 0

  constructor(private readonly options: ResilientSocketOptions) {}

  start() {
    if (!this.stopped) return
    this.stopped = false
    this.watchdogTimer = setInterval(
      () => this.checkHealth(),
      this.options.watchdogIntervalMs ?? 1000,
    )
    this.connect()
  }

  stop() {
    if (this.stopped) return
    this.stopped = true
    this.generation += 1
    this.clearTimers()
    const socket = this.socket
    this.socket = null
    if (socket) {
      socket.onopen = null
      socket.onmessage = null
      socket.onerror = null
      socket.onclose = null
      socket.close()
    }
  }

  send(payload: string) {
    if (this.socket?.readyState !== 1) return false
    this.socket.send(payload)
    return true
  }

  private connect() {
    if (this.stopped) return
    if (this.retryTimer) {
      clearTimeout(this.retryTimer)
      this.retryTimer = null
    }
    const generation = ++this.generation
    let socket: WebSocket
    try {
      socket = (this.options.socketFactory ?? ((url) => new WebSocket(url)))(
        this.options.url,
      )
    } catch {
      this.options.onDisconnect('connection-construction-failed')
      this.scheduleRetry()
      return
    }
    this.socket = socket
    this.lastMessageAt = (this.options.now ?? Date.now)()
    this.connectTimer = setTimeout(
      () => this.drop(generation, 'connection-timeout'),
      this.options.connectTimeoutMs ?? 8000,
    )

    socket.onopen = () => {
      if (!this.isCurrent(socket, generation)) return
      this.clearConnectTimer()
      this.attempt = 0
      this.lastMessageAt = (this.options.now ?? Date.now)()
      this.options.onOpen()
    }
    socket.onmessage = (event) => {
      if (!this.isCurrent(socket, generation)) return
      this.lastMessageAt = (this.options.now ?? Date.now)()
      this.options.onMessage(event)
    }
    socket.onerror = () => this.drop(generation, 'socket-error')
    socket.onclose = () => this.drop(generation, 'socket-closed')
  }

  private checkHealth() {
    if (this.stopped || this.socket?.readyState !== 1) return
    const elapsed = (this.options.now ?? Date.now)() - this.lastMessageAt
    if (elapsed > (this.options.staleTimeoutMs ?? 12_000)) {
      this.drop(this.generation, 'server-stale')
    }
  }

  private drop(generation: number, reason: string) {
    if (this.stopped || generation !== this.generation) return
    const socket = this.socket
    this.socket = null
    this.generation += 1
    this.clearConnectTimer()
    if (socket) {
      socket.onopen = null
      socket.onmessage = null
      socket.onerror = null
      socket.onclose = null
      socket.close()
    }
    this.options.onDisconnect(reason)
    this.scheduleRetry()
  }

  private scheduleRetry() {
    if (this.stopped || this.retryTimer) return
    this.attempt += 1
    const base = this.options.baseRetryMs ?? 750
    const maximum = this.options.maxRetryMs ?? 10_000
    const backoff = Math.min(maximum, base * 2 ** Math.min(6, this.attempt - 1))
    const jitter = 0.85 + (this.options.random ?? Math.random)() * 0.3
    const delay = Math.round(backoff * jitter)
    this.options.onRetry?.(delay, this.attempt)
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      this.connect()
    }, delay)
  }

  private isCurrent(socket: WebSocket, generation: number) {
    return !this.stopped && this.socket === socket && this.generation === generation
  }

  private clearConnectTimer() {
    if (!this.connectTimer) return
    clearTimeout(this.connectTimer)
    this.connectTimer = null
  }

  private clearTimers() {
    this.clearConnectTimer()
    if (this.retryTimer) clearTimeout(this.retryTimer)
    if (this.watchdogTimer) clearInterval(this.watchdogTimer)
    this.retryTimer = null
    this.watchdogTimer = null
  }
}
