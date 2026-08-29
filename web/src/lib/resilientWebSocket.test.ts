import { afterEach, describe, expect, it, vi } from 'vitest'
import { ResilientWebSocket } from './resilientWebSocket'

class FakeSocket {
  readyState = 0
  closed = 0
  sent: string[] = []
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null

  open() {
    this.readyState = 1
    this.onopen?.({} as Event)
  }

  message(data: string) {
    this.onmessage?.({ data } as MessageEvent)
  }

  serverClose() {
    this.readyState = 3
    this.onclose?.({} as CloseEvent)
  }

  fail() {
    this.onerror?.({} as Event)
  }

  close() {
    this.readyState = 3
    this.closed += 1
  }

  send(payload: string) {
    this.sent.push(payload)
  }
}

afterEach(() => {
  vi.useRealTimers()
})

describe('ResilientWebSocket', () => {
  it('reconnects repeatedly after a server close', () => {
    vi.useFakeTimers()
    const sockets: FakeSocket[] = []
    const disconnected: string[] = []
    const connection = new ResilientWebSocket({
      url: 'ws://example.test/ws',
      onOpen: () => undefined,
      onMessage: () => undefined,
      onDisconnect: (reason) => disconnected.push(reason),
      socketFactory: () => {
        const socket = new FakeSocket()
        sockets.push(socket)
        return socket as unknown as WebSocket
      },
      random: () => 0,
    })

    connection.start()
    sockets[0].open()
    sockets[0].serverClose()
    expect(disconnected).toEqual(['socket-closed'])
    vi.advanceTimersByTime(1000)
    expect(sockets).toHaveLength(2)
    sockets[1].open()
    sockets[1].serverClose()
    vi.advanceTimersByTime(1000)
    expect(sockets).toHaveLength(3)
    connection.stop()
  })

  it('recovers even when an error never produces a close event', () => {
    vi.useFakeTimers()
    const sockets: FakeSocket[] = []
    const connection = new ResilientWebSocket({
      url: 'ws://example.test/ws',
      onOpen: () => undefined,
      onMessage: () => undefined,
      onDisconnect: () => undefined,
      socketFactory: () => {
        const socket = new FakeSocket()
        sockets.push(socket)
        return socket as unknown as WebSocket
      },
      random: () => 0,
    })

    connection.start()
    sockets[0].fail()
    vi.advanceTimersByTime(1000)
    expect(sockets).toHaveLength(2)
    connection.stop()
  })

  it('reconnects a silently stale open connection', () => {
    vi.useFakeTimers()
    let now = 0
    const sockets: FakeSocket[] = []
    const disconnected: string[] = []
    const connection = new ResilientWebSocket({
      url: 'ws://example.test/ws',
      onOpen: () => undefined,
      onMessage: () => undefined,
      onDisconnect: (reason) => disconnected.push(reason),
      socketFactory: () => {
        const socket = new FakeSocket()
        sockets.push(socket)
        return socket as unknown as WebSocket
      },
      now: () => now,
      random: () => 0,
      staleTimeoutMs: 5000,
      watchdogIntervalMs: 1000,
    })

    connection.start()
    sockets[0].open()
    now = 6000
    vi.advanceTimersByTime(1000)
    expect(disconnected).toEqual(['server-stale'])
    vi.advanceTimersByTime(1000)
    expect(sockets).toHaveLength(2)
    connection.stop()
  })

  it('treats heartbeat messages as connection activity', () => {
    vi.useFakeTimers()
    let now = 0
    const socket = new FakeSocket()
    const connection = new ResilientWebSocket({
      url: 'ws://example.test/ws',
      onOpen: () => undefined,
      onMessage: () => undefined,
      onDisconnect: () => undefined,
      socketFactory: () => socket as unknown as WebSocket,
      now: () => now,
      staleTimeoutMs: 5000,
      watchdogIntervalMs: 1000,
    })

    connection.start()
    socket.open()
    now = 4000
    socket.message('{"type":"heartbeat"}')
    now = 8000
    vi.advanceTimersByTime(1000)
    expect(socket.closed).toBe(0)
    connection.stop()
  })
})
