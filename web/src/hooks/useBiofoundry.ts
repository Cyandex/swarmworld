import { useCallback, useEffect, useRef, useState } from 'react'
import { createDemoSnapshot, DEMO_CONTROL, evolveDemo, initialDemoEvents } from '../lib/demo'
import { dynamicsPointFromSnapshot, initialDynamicsHistory, mergeDynamicsPoint } from '../lib/dynamics'
import { mergeReplayFrames, shouldAdvanceDemo, shouldResetLiveTimeline } from '../lib/liveState'
import { ResilientWebSocket } from '../lib/resilientWebSocket'
import type { ClientCommand, ConnectionMode, ControlState, DynamicsHistory, DynamicsPoint, HistoryFrame, Snapshot, WorldEvent } from '../types'

const MAX_HISTORY = 256
const MAX_EVENTS = 2_000
const LEGACY_REPLAY_INTERVAL = 24

function defaultSocketUrl() {
  const configured = import.meta.env.VITE_BIOFOUNDRY_WS as string | undefined
  if (configured) return configured
  return 'ws://127.0.0.1:8765/ws'
}

export function useBiofoundry() {
  const initial = useRef(createDemoSnapshot())
  const [snapshot, setSnapshot] = useState<Snapshot>(initial.current)
  const [events, setEvents] = useState<WorldEvent[]>(initialDemoEvents(initial.current))
  const [control, setControl] = useState<ControlState>(DEMO_CONTROL)
  const [mode, setMode] = useState<ConnectionMode>('connecting')
  const [history, setHistory] = useState<HistoryFrame[]>([
    { snapshot: initial.current, events: initialDemoEvents(initial.current) },
  ])
  const [dynamicsHistory, setDynamicsHistory] = useState<DynamicsHistory>(() => initialDynamicsHistory(initial.current))
  const connectionRef = useRef<ResilientWebSocket | null>(null)
  const liveRef = useRef(false)
  const hasAuthoritativeSnapshotRef = useRef(false)
  const snapshotRef = useRef(initial.current)
  const controlRef = useRef(DEMO_CONTROL)

  useEffect(() => { snapshotRef.current = snapshot }, [snapshot])
  useEffect(() => { controlRef.current = control }, [control])

  const acceptFrame = useCallback((
    next: Snapshot,
    frameEvents: WorldEvent[],
    resetTimeline = false,
    replayCheckpoint?: boolean,
    replaceEvents = false,
  ) => {
    snapshotRef.current = next
    setSnapshot(next)
    if (resetTimeline) {
      setEvents(frameEvents.slice(-240))
      setHistory([{ snapshot: next, events: frameEvents }])
    } else if (replaceEvents) {
      setEvents(frameEvents.slice(-MAX_EVENTS))
    } else if (frameEvents.length) {
      setEvents((current) => [...current, ...frameEvents].slice(-MAX_EVENTS))
    }
    const shouldCheckpoint = replayCheckpoint
      ?? next.tick % LEGACY_REPLAY_INTERVAL === 0
    if (!resetTimeline && shouldCheckpoint) {
      setHistory((current) => mergeReplayFrames(
        current,
        [{ snapshot: next, events: frameEvents }],
        MAX_HISTORY,
      ))
    }
  }, [])

  useEffect(() => {
    const connection = new ResilientWebSocket({
      url: defaultSocketUrl(),
      onOpen: () => {
        liveRef.current = true
        // A TCP/WebSocket connection is not yet proof of an authoritative frame.
        // Keep the last real frame visible until the server's immediate snapshot arrives.
        setMode(hasAuthoritativeSnapshotRef.current ? 'reconnecting' : 'connecting')
      },
      onMessage: (message) => {
        try {
          const packet = JSON.parse(String(message.data)) as {
            type?: string
            snapshot?: Snapshot
            events?: WorldEvent[]
            control?: ControlState
            dynamics_history?: DynamicsHistory
            dynamics_point?: DynamicsPoint
            replay_checkpoint?: boolean
            playback_seek?: boolean
            frame?: HistoryFrame
          }
          if (packet.type === 'replay_frame' && packet.frame) {
            setHistory((current) => mergeReplayFrames(current, [packet.frame!], MAX_HISTORY))
            return
          }
          if (packet.control) {
            controlRef.current = packet.control
            setControl(packet.control)
          }
          if (packet.snapshot) {
            const playbackMode = Boolean(packet.control?.playback_mode ?? controlRef.current.playback_mode)
            const firstAuthoritative = !hasAuthoritativeSnapshotRef.current
            const resetTimeline = firstAuthoritative || (
              !playbackMode && shouldResetLiveTimeline(
                hasAuthoritativeSnapshotRef.current,
                snapshotRef.current,
                packet.snapshot,
              )
            )
            hasAuthoritativeSnapshotRef.current = true
            liveRef.current = true
            acceptFrame(
              packet.snapshot,
              packet.events ?? [],
              resetTimeline,
              packet.replay_checkpoint,
              Boolean(packet.playback_seek),
            )
            if (packet.dynamics_history?.points?.length) {
              setDynamicsHistory(packet.dynamics_history)
            } else if (packet.dynamics_point) {
              setDynamicsHistory((current) => mergeDynamicsPoint(current, packet.dynamics_point!))
            } else if (!playbackMode) {
              const fallbackPoint = dynamicsPointFromSnapshot(packet.snapshot)
              setDynamicsHistory((current) => resetTimeline
                ? { ...initialDynamicsHistory(packet.snapshot!), points: [fallbackPoint] }
                : mergeDynamicsPoint(current, fallbackPoint))
            }
            setMode(playbackMode ? 'playback' : packet.type === 'complete' ? 'complete' : 'live')
          } else if (packet.type === 'complete') {
            setMode('complete')
          }
        } catch {
          // A malformed presentation packet is ignored; the authoritative server is untouched.
        }
      },
      onDisconnect: () => {
        liveRef.current = false
        setMode('reconnecting')
      },
    })
    connectionRef.current = connection
    connection.start()
    return () => {
      liveRef.current = false
      connection.stop()
      if (connectionRef.current === connection) connectionRef.current = null
    }
  }, [acceptFrame])

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (!shouldAdvanceDemo(
        liveRef.current,
        hasAuthoritativeSnapshotRef.current,
        controlRef.current.paused,
      )) return
      const advanced = evolveDemo(snapshotRef.current)
      acceptFrame(advanced.snapshot, advanced.events, false, true)
      setDynamicsHistory((current) => mergeDynamicsPoint(current, dynamicsPointFromSnapshot(advanced.snapshot)))
    }, Math.max(120, 720 / control.speed_multiplier))
    return () => window.clearInterval(timer)
  }, [acceptFrame, control.speed_multiplier])

  const sendCommand = useCallback((command: ClientCommand) => {
    if (liveRef.current && connectionRef.current?.send(JSON.stringify(command))) {
      return
    }

    // Never mutate an authoritative server frame locally while reconnecting.
    if (hasAuthoritativeSnapshotRef.current) return

    if (command.command === 'toggle_pause') {
      setControl((current) => ({ ...current, paused: !current.paused }))
    } else if (command.command === 'set_paused') {
      setControl((current) => ({ ...current, paused: Boolean(command.paused) }))
    } else if (command.command === 'set_speed') {
      setControl((current) => ({ ...current, speed_multiplier: Math.max(0.25, Math.min(8, command.speed_multiplier ?? 1)) }))
    } else if (command.command === 'step') {
      const advanced = evolveDemo(snapshotRef.current)
      setControl((current) => ({ ...current, paused: true }))
      acceptFrame(advanced.snapshot, advanced.events, false, true)
      setDynamicsHistory((current) => mergeDynamicsPoint(current, dynamicsPointFromSnapshot(advanced.snapshot)))
    } else if (command.command === 'manual_action' && command.agent && command.action) {
      const index = snapshotRef.current.agents.ids.indexOf(command.agent)
      if (index < 0) return
      const next: Snapshot = structuredClone(snapshotRef.current)
      next.agents.last_action[index] = command.action.verb
      const direction = command.action.direction ?? 0
      const dx = [0, 0, 1, 0, -1][direction] ?? 0
      const dy = [0, -1, 0, 1, 0][direction] ?? 0
      if (command.action.verb === 1) {
        next.agents.x[index] = Math.max(0, Math.min(next.world.width - 1, next.agents.x[index] + dx))
        next.agents.y[index] = Math.max(0, Math.min(next.world.height - 1, next.agents.y[index] + dy))
      }
      next.tick += 1
      const manualEvent: WorldEvent = { tick: next.tick, kind: 'human_intervention', payload: { agent: command.agent, verb: command.action.verb } }
      acceptFrame(next, [manualEvent], false, true)
      setDynamicsHistory((current) => mergeDynamicsPoint(current, dynamicsPointFromSnapshot(next)))
    }
  }, [acceptFrame])

  return { snapshot, events, control, mode, history, dynamicsHistory, sendCommand, socketUrl: defaultSocketUrl() }
}
