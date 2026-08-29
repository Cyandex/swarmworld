import {
  Activity, Atom, Beaker, BookOpen, Box, BrainCircuit, Camera, ChevronDown, CircleDot,
  FlaskConical, Gauge, GripVertical, Hammer, Info, Layers3, Leaf, Microscope, MoveDown,
  MoveLeft, MoveRight, MoveUp, MapPin, Network, Pause, Play, Radio, RotateCcw,
  ScanSearch, Search, Sparkles, Square, StepForward, TestTube2, Trash2, Video,
  Waves, Wrench, Zap,
} from 'lucide-react'
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties, KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from 'react'
import { ACTION_NAMES, ARTIFACT_NAMES, FIELD_LABELS, RESOURCE_NAMES, STATION_NAMES, TERRAIN_NAMES } from './constants'
import { MetricChart } from './components/MetricChart'
import { PanelErrorBoundary } from './components/PanelErrorBoundary'
import { WorldScene } from './components/WorldScene'
import { useBiofoundry } from './hooks/useBiofoundry'
import { useVideoExport } from './hooks/useVideoExport'
import type { LineageGraphLayout, LineageNodeDetail } from './lib/lineage'
import {
  buildLineageNetworkPaperArtifact,
  buildPanelPaperArtifact,
  buildWorldPaperArtifact,
  downloadPaperArtifact,
} from './lib/paperExport'
import type { FieldMode, Snapshot, WorldEvent } from './types'

type PanelTab = 'inspect' | 'activity' | 'lineage' | 'report'
type ActivityFilter = 'all' | 'messages' | 'science' | 'artifacts' | 'errors'

const NetworkView = lazy(() => import('./components/NetworkView').then((module) => ({ default: module.NetworkView })))

const OBSERVER_DEFAULT_WIDTH = 348
const OBSERVER_MIN_WIDTH = 320
const OBSERVER_WIDE_WIDTH = 640
const OBSERVER_MAX_WIDTH = 920
const OBSERVER_WIDTH_STORAGE_KEY = 'biofoundry-observer-width'

function clampObserverWidth(width: number) {
  const viewportMaximum = typeof window === 'undefined'
    ? OBSERVER_MAX_WIDTH
    : Math.max(OBSERVER_MIN_WIDTH, window.innerWidth - 260)
  return Math.round(Math.max(
    OBSERVER_MIN_WIDTH,
    Math.min(OBSERVER_MAX_WIDTH, viewportMaximum, width),
  ))
}

function initialObserverWidth() {
  if (typeof window === 'undefined') return OBSERVER_DEFAULT_WIDTH
  const stored = Number(window.localStorage.getItem(OBSERVER_WIDTH_STORAGE_KEY))
  return clampObserverWidth(Number.isFinite(stored) && stored > 0 ? stored : OBSERVER_DEFAULT_WIDTH)
}

function mean(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length)
}

function compactAgent(id: string) {
  return id.replace('agent_', 'A')
}

function eventSummary(event: WorldEvent) {
  const payload = event.payload
  switch (event.kind) {
    case 'agent_deliberated': return `${compactAgent(String(payload.agent ?? 'agent'))} chose ${ACTION_NAMES[Number(payload.verb ?? 0)] ?? 'an action'}`
    case 'model_error': return `Model request failed for ${compactAgent(String(payload.agent ?? 'agent'))}`
    case 'agents_moved': return `${((payload.indices as number[] | undefined) ?? []).length} agents changed position`
    case 'resource_harvested': return `${compactAgent(String(payload.agent ?? 'agent'))} harvested ${RESOURCE_NAMES[Number(payload.resource ?? 0)] ?? 'material'}`
    case 'artifact_built': return `${compactAgent(String(payload.agent ?? 'agent'))} built ${String(payload.artifact_name ?? ARTIFACT_NAMES[Number(payload.artifact_type ?? 0)] ?? 'an artifact')}`
    case 'artifact_program_installed': return `${compactAgent(String(payload.agent ?? 'agent'))} installed an artifact program`
    case 'insight_deposited': return `${compactAgent(String(payload.author ?? 'agent'))} deposited an insight`
    case 'message_delivered': return `${compactAgent(String(payload.sender ?? 'agent'))} → ${((payload.recipients as string[] | undefined) ?? []).map(compactAgent).join(', ') || 'nearby agents'}`
    case 'action_result': return `${compactAgent(String(payload.agent ?? 'agent'))} ${String(payload.action ?? ACTION_NAMES[Number(payload.verb ?? 0)] ?? 'acted').toLowerCase()} ${payload.success === false ? 'failed' : 'succeeded'}`
    case 'sample_tested': return `${compactAgent(String(payload.agent ?? 'agent'))} completed a material test`
    case 'human_intervention': return `Human directed ${compactAgent(String(payload.agent ?? 'agent'))}`
    default: return event.kind.replaceAll('_', ' ')
  }
}

function eventCategory(event: WorldEvent): ActivityFilter {
  if (event.kind === 'model_error' || event.kind === 'action_rejected' || event.payload.success === false) return 'errors'
  if (event.kind === 'message_delivered') return 'messages'
  if (event.kind === 'action_result') {
    const verb = Number(event.payload.verb ?? -1)
    if ([11, 17, 18].includes(verb)) return 'messages'
    if ([8, 9, 10, 14, 15, 19].includes(verb)) return 'artifacts'
    if ([2, 5, 6, 7, 12, 13].includes(verb)) return 'science'
  }
  if (event.kind.includes('artifact') || event.kind.includes('program')) return 'artifacts'
  if (
    event.kind.includes('test') || event.kind.includes('sample') || event.kind.includes('recipe')
    || event.kind.includes('insight') || event.kind.includes('research')
  ) return 'science'
  return 'all'
}

function eventDetail(event: WorldEvent) {
  if (event.kind === 'message_delivered') return String(event.payload.message ?? '')
  if (event.kind === 'action_result') {
    return String(event.payload.intent ?? event.payload.detail ?? '')
  }
  if (event.kind === 'action_rejected') return String(event.payload.reason ?? '')
  return String(event.payload.detail ?? event.payload.claim ?? '')
}

function MetricPill({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div className="metric-pill">
      <span>{label}</span>
      <strong style={{ color: accent }}>{value}</strong>
    </div>
  )
}

function PaperSnapshotButton({
  label,
  text: buttonText,
  className = '',
  icon = 'camera',
  disabled = false,
  onExport,
}: {
  label: string
  text: string
  className?: string
  icon?: 'camera' | 'network'
  disabled?: boolean
  onExport: () => Promise<void>
}) {
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState('')

  const runExport = async () => {
    if (exporting) return
    setExporting(true)
    setError('')
    try {
      await onExport()
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : 'Snapshot export failed'
      setError(message)
      console.error('Unable to export paper snapshot', reason)
    } finally {
      setExporting(false)
    }
  }

  return (
    <button
      className={`paper-snapshot-button ${className}`}
      aria-label={label}
      title={error || `${label} as PNG and SVG on white`}
      disabled={exporting || disabled}
      onClick={runExport}
    >{icon === 'network' ? <Network size={13} /> : <Camera size={13} />}<span>{exporting ? 'Rendering…' : buttonText}</span></button>
  )
}

function SituationReport({ snapshot, events }: { snapshot: Snapshot; events: WorldEvent[] }) {
  const actionCounts = snapshot.agents.last_action.reduce<Record<number, number>>((counts, action) => {
    counts[action] = (counts[action] ?? 0) + 1
    return counts
  }, {})
  const [dominantAction, dominantCount] = Object.entries(actionCounts).sort((a, b) => b[1] - a[1])[0] ?? ['0', 0]
  const recentCommunications = events.slice(-80).filter((event) => event.kind === 'message_delivered').length
  const recentDiscoveries = events.slice(-80).filter((event) => event.kind.includes('insight') || event.kind.includes('artifact')).length
  const healthyArtifacts = snapshot.artifacts.health.filter((health) => health >= 0.7).length
  const moisture = mean(snapshot.world.moisture)
  const contamination = mean(snapshot.world.contamination)
  const energy = mean(snapshot.agents.energy)

  return (
    <div className="report-copy">
      <div className="report-kicker"><Sparkles size={13} /> Deterministic situation report</div>
      <h2>{snapshot.research?.completed ? 'The collective discovery mission succeeded.' : 'The society is operational and materially exploratory.'}</h2>
      <p>
        At tick <b>{snapshot.tick.toLocaleString()}</b>, {snapshot.agents.count} agents maintain {snapshot.artifacts.count} active artifacts.
        Mean agent energy is <b>{Math.round(energy * 100)}%</b>; {healthyArtifacts} artifacts are above the 70% health threshold.
      </p>
      <div className="report-grid">
        <div><span>Environmental state</span><strong>{Math.round(moisture * 100)}% moisture</strong><small>{Math.round(contamination * 1000) / 10}% mean contamination</small></div>
        <div><span>Collective behavior</span><strong>{ACTION_NAMES[Number(dominantAction)]}</strong><small>{dominantCount} agents currently share this action</small></div>
        <div><span>Recent knowledge flow</span><strong>{recentCommunications} exchanges</strong><small>{recentDiscoveries} discovery-related events in the event window</small></div>
        <div><span>Material utility</span><strong>{snapshot.metrics.artifact_score.toFixed(2)}</strong><small>{snapshot.metrics.archive_entries} archive records · {snapshot.metrics.deposited_insights} insights</small></div>
        {snapshot.research && <div><span>Research progress</span><strong>{Math.round(snapshot.research.score * 100)}%</strong><small>{snapshot.research.outcome_success ? 'functional outcome achieved' : 'scientific cycle in progress'}</small></div>}
      </div>
      <p className="report-note">
        This report is calculated from the displayed state and event log. It does not call an LLM or change agent behavior.
      </p>
    </div>
  )
}

function App() {
  const { snapshot: liveSnapshot, events, control, mode, history, dynamicsHistory, sendCommand, socketUrl } = useBiofoundry()
  const {
    recording: videoRecording,
    error: videoError,
    start: startVideoExport,
    stop: stopVideoExport,
  } = useVideoExport()
  const [field, setField] = useState<FieldMode>('terrain')
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)
  const [selectedArtifact, setSelectedArtifact] = useState<string | null>(null)
  const [selectedTile, setSelectedTile] = useState<[number, number] | null>(null)
  const [selectedLineageNode, setSelectedLineageNode] = useState<LineageNodeDetail | null>(null)
  const [lineageGraphLayout, setLineageGraphLayout] = useState<LineageGraphLayout | null>(null)
  const [objectQuery, setObjectQuery] = useState('')
  const [showResources, setShowResources] = useState(true)
  const [showSignals, setShowSignals] = useState(true)
  const [showTrails, setShowTrails] = useState(true)
  const [tab, setTab] = useState<PanelTab>('inspect')
  const [replayIndex, setReplayIndex] = useState(-1)
  const [analyticsOpen, setAnalyticsOpen] = useState(true)
  const [activityFilter, setActivityFilter] = useState<ActivityFilter>('all')
  const [observerWidth, setObserverWidth] = useState(initialObserverWidth)
  const [observerResizing, setObserverResizing] = useState(false)
  const observerResizeRef = useRef<{ startX: number; startWidth: number } | null>(null)

  useEffect(() => {
    window.localStorage.setItem(OBSERVER_WIDTH_STORAGE_KEY, String(observerWidth))
  }, [observerWidth])

  useEffect(() => {
    const handleResize = () => setObserverWidth((current) => clampObserverWidth(current))
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const startObserverResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    observerResizeRef.current = { startX: event.clientX, startWidth: observerWidth }
    event.currentTarget.setPointerCapture(event.pointerId)
    setObserverResizing(true)
  }

  const moveObserverResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    const resize = observerResizeRef.current
    if (!resize) return
    setObserverWidth(clampObserverWidth(resize.startWidth + resize.startX - event.clientX))
  }

  const stopObserverResize = () => {
    observerResizeRef.current = null
    setObserverResizing(false)
  }

  const resizeObserverFromKeyboard = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    if (event.key === 'ArrowLeft') setObserverWidth((current) => clampObserverWidth(current + 24))
    if (event.key === 'ArrowRight') setObserverWidth((current) => clampObserverWidth(current - 24))
    if (event.key === 'Home') setObserverWidth(OBSERVER_DEFAULT_WIDTH)
    if (event.key === 'End') setObserverWidth(clampObserverWidth(OBSERVER_WIDE_WIDTH))
  }

  const toggleWideObserver = () => {
    setObserverWidth((current) => clampObserverWidth(
      current < OBSERVER_WIDE_WIDTH - 40 ? OBSERVER_WIDE_WIDTH : OBSERVER_DEFAULT_WIDTH,
    ))
  }

  const displayedFrame = replayIndex >= 0 ? history[replayIndex] : null
  const snapshot = displayedFrame?.snapshot ?? liveSnapshot
  const scenario = snapshot.scenario ?? snapshot.world.scenario
  const catalogName = (kind: string, index: number, fallback: string[]) => (
    scenario?.catalogs?.[kind]?.[index]?.name
    ?? scenario?.catalogs?.[kind]?.[index]?.id
    ?? fallback[index]
    ?? 'Unknown'
  )
  const fieldOptions = scenario?.fields?.length
    ? [{ id: 'terrain', name: 'Terrain' }, ...scenario.fields.map((item) => ({ id: item.id, name: item.name ?? item.id.replaceAll('_', ' ') }))]
    : (Object.keys(FIELD_LABELS) as FieldMode[]).map((id) => ({ id, name: FIELD_LABELS[id as keyof typeof FIELD_LABELS] }))
  const displayedEvents = replayIndex >= 0
    ? history.slice(0, replayIndex + 1).flatMap((frame) => frame.events).slice(-240)
    : events

  const selectedAgentIndex = selectedAgent ? snapshot.agents.ids.indexOf(selectedAgent) : -1
  const selectedArtifactIndex = selectedArtifact ? snapshot.artifacts.ids.indexOf(selectedArtifact) : -1
  const selectedTileIndex = selectedTile
    ? selectedTile[1] * snapshot.world.width + selectedTile[0]
    : -1
  const live = mode === 'live' || mode === 'complete'
  const playbackMode = Boolean(control.playback_mode)
  const connected = live || mode === 'playback'

  useEffect(() => {
    if (videoRecording && playbackMode && control.playback_complete) {
      stopVideoExport()
    }
  }, [control.playback_complete, playbackMode, stopVideoExport, videoRecording])

  const selectedTileAgents = selectedTile ? snapshot.agents.ids.flatMap((id, index) => (
    snapshot.agents.x[index] === selectedTile[0] && snapshot.agents.y[index] === selectedTile[1] ? [{ id, index }] : []
  )) : []
  const selectedTileArtifacts = selectedTile ? snapshot.artifacts.ids.flatMap((id, index) => (
    snapshot.artifacts.x[index] === selectedTile[0] && snapshot.artifacts.y[index] === selectedTile[1] ? [{ id, index }] : []
  )) : []
  const worldObjects = useMemo(() => {
    const query = objectQuery.trim().toLowerCase()
    const matches: Array<{ id: string; label: string; kind: 'artifact' | 'agent'; x: number; y: number; detail: string }> = []
    snapshot.artifacts.ids.forEach((id, index) => {
      const label = snapshot.artifacts.name[index] || id
      if (!query || `${id} ${label}`.toLowerCase().includes(query)) {
        matches.push({ id, label, kind: 'artifact', x: snapshot.artifacts.x[index], y: snapshot.artifacts.y[index], detail: `performance ${snapshot.artifacts.performance[index].toFixed(3)}` })
      }
    })
    snapshot.agents.ids.forEach((id, index) => {
      const label = compactAgent(id)
      if (!query || `${id} ${label}`.toLowerCase().includes(query)) {
        matches.push({ id, label, kind: 'agent', x: snapshot.agents.x[index], y: snapshot.agents.y[index], detail: ACTION_NAMES[snapshot.agents.last_action[index]] ?? 'Unknown action' })
      }
    })
    return matches.slice(0, 80)
  }, [objectQuery, snapshot])

  const trail = useMemo(() => {
    if (!selectedAgent) return []
    return history.filter((frame) => frame.snapshot.tick <= snapshot.tick).flatMap((frame) => {
      const index = frame.snapshot.agents.ids.indexOf(selectedAgent)
      if (index < 0) return []
      return [[
        frame.snapshot.agents.x[index] - frame.snapshot.world.width / 2,
        0.2,
        frame.snapshot.agents.y[index] - frame.snapshot.world.height / 2,
      ] as [number, number, number]]
    })
  }, [history, selectedAgent])

  const actionLeaders = useMemo(() => {
    const counts = snapshot.agents.last_action.reduce<Record<number, number>>((accumulator, action) => {
      accumulator[action] = (accumulator[action] ?? 0) + 1
      return accumulator
    }, {})
    return Object.entries(counts)
      .map(([action, count]) => ({ action: Number(action), count }))
      .filter(({ action }) => action > 0)
      .sort((left, right) => right.count - left.count)
      .slice(0, 5)
  }, [snapshot.agents.last_action])

  const filteredActivity = useMemo(() => displayedEvents.filter((event) => (
    activityFilter === 'all' || eventCategory(event) === activityFilter
  )), [activityFilter, displayedEvents])

  const worldHighlights = useMemo(() => displayedEvents.filter((event) => {
    const age = snapshot.tick - event.tick
    return age >= 0 && age <= 4 && [
      'message_delivered', 'artifact_built', 'artifact_program_installed',
      'sample_tested', 'insight_deposited', 'action_rejected', 'action_result',
    ].includes(event.kind)
  }).slice(-4).reverse(), [displayedEvents, snapshot.tick])

  const averageEnergy = mean(snapshot.agents.energy)
  const averageMoisture = mean(snapshot.world.moisture)
  const averageContamination = mean(snapshot.world.contamination)

  const selectAgent = (id: string) => {
    setSelectedAgent(id)
    setSelectedArtifact(null)
    setSelectedTile(null)
    setTab('inspect')
  }
  const selectArtifact = (id: string) => {
    setSelectedArtifact(id)
    setSelectedAgent(null)
    setSelectedTile(null)
    setTab('inspect')
  }
  const selectTile = (x: number, y: number) => {
    setSelectedTile([x, y])
    setSelectedAgent(null)
    setSelectedArtifact(null)
    setTab('inspect')
  }
  const clearSelection = () => {
    setSelectedAgent(null)
    setSelectedArtifact(null)
    setSelectedTile(null)
  }

  const directAgent = (verb: number, direction = 0) => {
    if (!selectedAgent || replayIndex >= 0 || playbackMode) return
    sendCommand({
      command: 'manual_action',
      agent: selectedAgent,
      action: {
        verb,
        direction,
        target_x: selectedAgentIndex >= 0 ? snapshot.agents.x[selectedAgentIndex] : -1,
        target_y: selectedAgentIndex >= 0 ? snapshot.agents.y[selectedAgentIndex] : -1,
      },
    })
  }

  const selectEventSubject = (event: WorldEvent) => {
    const agent = String(event.payload.agent ?? event.payload.sender ?? event.payload.author ?? '')
    const artifact = String(event.payload.artifact_id ?? event.payload.artifact ?? '')
    if (artifact && snapshot.artifacts.ids.includes(artifact)) selectArtifact(artifact)
    else if (agent && snapshot.agents.ids.includes(agent)) selectAgent(agent)
  }

  const toggleVideoExport = async () => {
    if (videoRecording) {
      stopVideoExport()
      return
    }
    try {
      await startVideoExport(`swarmworld-seed-${snapshot.seed}-n-${snapshot.agents.count}`)
      if (playbackMode) sendCommand({ command: 'play_from_start' })
    } catch (reason) {
      console.error('Unable to start video export', reason)
    }
  }

  const seekToHistoryIndex = (index: number) => {
    if (!playbackMode) {
      setReplayIndex(index)
      return
    }
    const tick = history[index]?.snapshot.tick
    if (tick === undefined) return
    setReplayIndex(-1)
    sendCommand({ command: 'seek', tick })
  }

  const playbackHistoryIndex = playbackMode
    ? history.reduce((best, frame, index) => (
      frame.snapshot.tick <= snapshot.tick && frame.snapshot.tick >= history[best].snapshot.tick
        ? index
        : best
    ), 0)
    : Math.max(0, history.length - 1)
  const timelineValue = replayIndex >= 0 ? replayIndex : playbackHistoryIndex
  const exportWorldSnapshot = () => downloadPaperArtifact(buildWorldPaperArtifact(
    snapshot,
    field,
    selectedAgent,
    selectedArtifact,
    selectedTile,
  ))
  const exportPanelSnapshot = () => downloadPaperArtifact(buildPanelPaperArtifact({
    tab,
    snapshot,
    events: displayedEvents,
    selectedAgentIndex,
    selectedArtifactIndex,
    selectedTile,
    selectedLineageNode,
  }))
  const exportLineageGraphSnapshot = async () => {
    if (!lineageGraphLayout) throw new Error('Open Lineage and wait for the graph layout')
    await downloadPaperArtifact(buildLineageNetworkPaperArtifact({
      tab: 'lineage',
      snapshot,
      events: displayedEvents,
      selectedAgentIndex,
      selectedArtifactIndex,
      selectedTile,
      selectedLineageNode,
    }, lineageGraphLayout))
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><Leaf size={17} /><Atom size={12} /></div>
          <div><strong>BIOFOUNDRY</strong><span>SOCIETY OBSERVATORY</span></div>
        </div>

        <div className="run-identity">
          <div className={`connection-dot ${control.provider_outage ? 'is-outage' : connected ? 'is-live' : 'is-demo'}`} />
          <div>
            <span>{control.provider_outage ? 'PROVIDER RETRY — WORLD PAUSED' : playbackMode ? (control.playback_complete ? 'OFFLINE REPLAY COMPLETE' : 'OFFLINE TRACE PLAYBACK') : live ? (mode === 'complete' ? 'EPISODE COMPLETE' : 'LIVE SIMULATION') : mode === 'connecting' ? 'CONNECTING' : mode === 'reconnecting' ? 'RECONNECTING' : 'DEMONSTRATION MODE'}</span>
            <small>{control.provider_outage ? `${control.model} · tick and action queues preserved` : playbackMode ? `${control.trace_name ?? 'recorded trace'} · no model calls` : live ? control.model : mode === 'reconnecting' ? `Retrying ${socketUrl}` : `No server at ${socketUrl}`}</small>
          </div>
        </div>

        <div className="transport-controls" aria-label="Simulation controls">
          <button
            className="icon-button primary"
            aria-label={control.paused ? 'Resume simulation' : 'Pause simulation'}
            title={control.paused ? 'Resume' : 'Pause'}
            disabled={replayIndex >= 0 && !playbackMode}
            onClick={() => sendCommand({ command: 'toggle_pause' })}
          >{control.paused ? <Play size={15} /> : <Pause size={15} />}</button>
          <button className="icon-button" aria-label="Advance one simulation tick" title="Step one tick" disabled={replayIndex >= 0 && !playbackMode} onClick={() => sendCommand({ command: 'step' })}><StepForward size={15} /></button>
          <label className="speed-control">
            <span>Speed</span>
            <select value={control.speed_multiplier} disabled={replayIndex >= 0 && !playbackMode} onChange={(event) => sendCommand({ command: 'set_speed', speed_multiplier: Number(event.target.value) })}>
              {(playbackMode ? [0.25, 0.5, 1, 2, 4, 8, 16, 32] : [0.25, 0.5, 1, 2, 4, 8]).map((speed) => <option key={speed} value={speed}>{speed}×</option>)}
            </select>
          </label>
          <button
            className={`icon-button video-button ${videoRecording ? 'recording' : ''}`}
            aria-label={videoRecording ? 'Stop and download video' : playbackMode ? 'Record complete replay' : 'Record observatory video'}
            title={videoError || (videoRecording ? 'Stop and download video' : playbackMode ? 'Choose this browser tab; playback will restart automatically' : 'Record this observatory tab')}
            disabled={!connected}
            onClick={toggleVideoExport}
          >{videoRecording ? <Square size={13} /> : <Video size={15} />}</button>
        </div>

        <div className="top-metrics">
          <MetricPill label="TICK" value={snapshot.tick.toLocaleString()} />
          <MetricPill label="AGENTS" value={snapshot.agents.count.toLocaleString()} accent="#72dcc0" />
          <MetricPill label="ARTIFACTS" value={snapshot.artifacts.count.toLocaleString()} accent="#efc968" />
          <MetricPill label="UTILITY" value={snapshot.metrics.artifact_score.toFixed(2)} accent="#efc968" />
          {snapshot.research && <MetricPill label="RESEARCH" value={`${Math.round(snapshot.research.score * 100)}%`} accent={snapshot.research.completed ? '#72dcc0' : '#8fb5ff'} />}
        </div>
      </header>

      <main
        className="world-stage"
        style={{ '--observer-width': `${observerWidth}px` } as CSSProperties}
      >
        <div className="canvas-wrap">
          <WorldScene
            snapshot={snapshot}
            events={displayedEvents}
            field={field}
            selectedAgent={selectedAgent}
            selectedArtifact={selectedArtifact}
            selectedTile={selectedTile}
            showResources={showResources}
            showSignals={showSignals}
            showTrails={showTrails}
            trail={trail}
            onSelectAgent={selectAgent}
            onSelectArtifact={selectArtifact}
            onSelectTile={selectTile}
            onClearSelection={clearSelection}
          />
          <div className="canvas-vignette" />
          <div className="camera-hint"><ScanSearch size={13} /> Click any object or tile to inspect · drag to orbit · scroll to zoom</div>
          {worldHighlights.length > 0 && (
            <div className="world-activity-overlay" aria-label="Recent world interactions">
              {worldHighlights.map((event, index) => (
                <button key={`${event.tick}-${event.kind}-${index}`} onClick={() => selectEventSubject(event)}>
                  <span className={`interaction-mark ${eventCategory(event)}`} />
                  <span><strong>{eventSummary(event)}</strong>{eventDetail(event) && <small>{eventDetail(event)}</small>}</span>
                  <time>T{event.tick}</time>
                </button>
              ))}
            </div>
          )}
          <PaperSnapshotButton
            className="world-export-button"
            label="Export world state"
            text="Paper snapshot"
            onExport={exportWorldSnapshot}
          />
          {replayIndex >= 0 && <div className="replay-badge"><RotateCcw size={13} /> REPLAYING TICK {snapshot.tick}</div>}
          {videoRecording && <div className="video-recording-badge"><span /> RECORDING COMPLETE OBSERVATORY</div>}
        </div>

        <aside className="view-controls glass-panel">
          <div className="panel-heading"><Layers3 size={14} /><span>World layers</span></div>
          <div className="field-selector">
            {fieldOptions.map((option) => (
              <button key={option.id} className={field === option.id ? 'active' : ''} onClick={() => setField(option.id)}>
                <span className={`field-swatch ${option.id}`} />{option.name}
              </button>
            ))}
          </div>
          <div className="panel-rule" />
          <label className="toggle-row"><input type="checkbox" checked={showResources} onChange={(event) => setShowResources(event.target.checked)} /><span /><Leaf size={13} /> {scenario ? 'Scenario resources' : 'Biological resources'}</label>
          <label className="toggle-row"><input type="checkbox" checked={showSignals} onChange={(event) => setShowSignals(event.target.checked)} /><span /><Radio size={13} /> Local exchanges</label>
          <label className="toggle-row"><input type="checkbox" checked={showTrails} onChange={(event) => setShowTrails(event.target.checked)} /><span /><Waves size={13} /> Selected trajectory</label>
          <div className="environment-mini">
            <div><small>MOISTURE</small><strong>{Math.round(averageMoisture * 100)}%</strong></div>
            <div><small>CONTAM.</small><strong>{(averageContamination * 100).toFixed(1)}%</strong></div>
          </div>
        </aside>

        <aside className={`observer-panel glass-panel ${observerResizing ? 'is-resizing' : ''}`}>
          <div
            className="observer-resize-handle"
            role="separator"
            aria-label="Resize observatory panel"
            aria-orientation="vertical"
            aria-valuemin={OBSERVER_MIN_WIDTH}
            aria-valuemax={OBSERVER_MAX_WIDTH}
            aria-valuenow={observerWidth}
            tabIndex={0}
            title="Drag to resize · double-click for wide lineage view"
            onPointerDown={startObserverResize}
            onPointerMove={moveObserverResize}
            onPointerUp={stopObserverResize}
            onPointerCancel={stopObserverResize}
            onLostPointerCapture={stopObserverResize}
            onKeyDown={resizeObserverFromKeyboard}
            onDoubleClick={toggleWideObserver}
          ><GripVertical size={13} /></div>
          <nav className={`panel-tabs ${tab === 'lineage' ? 'has-lineage-export' : ''}`} aria-label="Observatory panels">
            <button className={tab === 'inspect' ? 'active' : ''} onClick={() => setTab('inspect')}><Microscope size={14} />Inspect</button>
            <button className={tab === 'activity' ? 'active' : ''} onClick={() => setTab('activity')}><Activity size={14} />Activity</button>
            <button className={tab === 'lineage' ? 'active' : ''} onClick={() => setTab('lineage')}><Network size={14} />Lineage</button>
            <button className={tab === 'report' ? 'active' : ''} onClick={() => setTab('report')}><BookOpen size={14} />Report</button>
            <PaperSnapshotButton
              className="panel-export-button"
              label={`Export ${tab} panel`}
              text={tab === 'lineage' ? 'Lanes' : 'Snapshot'}
              onExport={exportPanelSnapshot}
            />
            {tab === 'lineage' && (
              <PaperSnapshotButton
                className="panel-export-button graph-export-button"
                label="Export force-directed lineage graph"
                text="Graph"
                icon="network"
                disabled={!lineageGraphLayout}
                onExport={exportLineageGraphSnapshot}
              />
            )}
          </nav>

          <div className="panel-content">
            {tab === 'inspect' && selectedAgentIndex >= 0 && (
              <div className="inspector">
                <div className="selection-title agent-selection">
                  <div className="selection-icon"><CircleDot size={18} /></div>
                  <div><small>SELECTED AGENT</small><h2>{compactAgent(selectedAgent!)}</h2><span>{selectedAgent}</span></div>
                </div>
                <div className="property-grid">
                  <div><span>Position</span><strong>{snapshot.agents.x[selectedAgentIndex]}, {snapshot.agents.y[selectedAgentIndex]}</strong></div>
                  <div><span>Energy</span><strong>{Math.round(snapshot.agents.energy[selectedAgentIndex] * 100)}%</strong></div>
                  <div><span>Inventory</span><strong>{snapshot.agents.inventory_total[selectedAgentIndex].toFixed(2)}</strong></div>
                  <div><span>Action</span><strong>{ACTION_NAMES[snapshot.agents.last_action[selectedAgentIndex]] ?? 'Unknown'}</strong></div>
                </div>
                <div className="local-context">
                  <span>LOCAL OBSERVATION</span>
                  <div className="context-row"><Leaf size={13} /><p>{catalogName('terrains', snapshot.world.terrain[snapshot.agents.y[selectedAgentIndex] * snapshot.world.width + snapshot.agents.x[selectedAgentIndex]], TERRAIN_NAMES)}</p></div>
                  <div className="context-row"><Beaker size={13} /><p>{catalogName('resources', snapshot.world.resource_kind[snapshot.agents.y[selectedAgentIndex] * snapshot.world.width + snapshot.agents.x[selectedAgentIndex]], RESOURCE_NAMES)}</p></div>
                  <small>The renderer exposes only observations broadcast by the scientific protocol.</small>
                </div>
                <div className="manual-control">
                  <div className="section-label"><MoveUp size={13} /> Human intervention</div>
                  <div className="direction-pad">
                    <button aria-label="Move north" onClick={() => directAgent(1, 1)}><MoveUp size={15} /></button>
                    <button aria-label="Move west" onClick={() => directAgent(1, 4)}><MoveLeft size={15} /></button>
                    <button aria-label="Inspect" className="center" onClick={() => directAgent(2)}><ScanSearch size={15} /></button>
                    <button aria-label="Move east" onClick={() => directAgent(1, 2)}><MoveRight size={15} /></button>
                    <button aria-label="Move south" onClick={() => directAgent(1, 3)}><MoveDown size={15} /></button>
                  </div>
                  <div className="action-buttons">
                    <button onClick={() => directAgent(3)}><Leaf size={13} />Harvest</button>
                    <button onClick={() => directAgent(5)}><Zap size={13} />Operate</button>
                    <button onClick={() => directAgent(9)}><Wrench size={13} />Repair</button>
                    <button className="danger" onClick={() => directAgent(10)}><Trash2 size={13} />Dismantle</button>
                  </div>
                  {replayIndex >= 0 && <small className="intervention-disabled">Return to live state to intervene.</small>}
                </div>
              </div>
            )}

            {tab === 'inspect' && selectedArtifactIndex >= 0 && (
              <div className="inspector">
                <div className="selection-title artifact-selection">
                  <div className="selection-icon"><Box size={18} /></div>
                  <div><small>AGENT-INVENTED MATERIAL SYSTEM</small><h2>{snapshot.artifacts.name[selectedArtifactIndex] || 'Untitled material system'}</h2><span>{selectedArtifact}</span></div>
                </div>
                <div className="property-grid">
                  <div><span>Position</span><strong>{snapshot.artifacts.x[selectedArtifactIndex]}, {snapshot.artifacts.y[selectedArtifactIndex]}</strong></div>
                  <div><span>Creator</span><strong>{snapshot.artifacts.creator?.[selectedArtifactIndex] ? compactAgent(snapshot.artifacts.creator[selectedArtifactIndex]) : 'Unknown'}</strong></div>
                  <div><span>Created</span><strong>{snapshot.artifacts.created_tick?.[selectedArtifactIndex] === undefined ? 'Unknown tick' : `T${snapshot.artifacts.created_tick[selectedArtifactIndex]}`}</strong></div>
                  <div><span>Causal parents</span><strong>{snapshot.artifacts.causal_parents?.[selectedArtifactIndex]?.length ?? 0}</strong></div>
                  <div><span>Health</span><strong>{Math.round(snapshot.artifacts.health[selectedArtifactIndex] * 100)}%</strong></div>
                  <div><span>Maturity</span><strong>{Math.round(snapshot.artifacts.maturity[selectedArtifactIndex] * 100)}%</strong></div>
                  <div><span>Performance</span><strong>{snapshot.artifacts.performance[selectedArtifactIndex].toFixed(3)}</strong></div>
                  <div><span>Peak since program</span><strong>{snapshot.artifacts.peak_performance[selectedArtifactIndex].toFixed(3)}</strong></div>
                  <div><span>Lifetime peak</span><strong>{(snapshot.artifacts.lifetime_peak_performance?.[selectedArtifactIndex] ?? snapshot.artifacts.peak_performance[selectedArtifactIndex]).toFixed(3)}</strong></div>
                  <div><span>Storage</span><strong>{snapshot.artifacts.storage[selectedArtifactIndex].toFixed(3)}</strong></div>
                </div>
                <div className="program-card">
                  <div><Sparkles size={15} /><span>SCIENTIFIC HYPOTHESIS</span></div>
                  <p>{snapshot.artifacts.claimed_function[selectedArtifactIndex] || 'No claimed function recorded.'}</p>
                  <p>{snapshot.artifacts.architecture[selectedArtifactIndex] || 'No architecture description recorded.'}</p>
                  <small>Inspired by {(snapshot.artifacts.bio_inspiration[selectedArtifactIndex] ?? []).join(', ') || 'unspecified biological systems'}. Prose claims do not alter physics.</small>
                </div>
                <div className="program-card">
                  <div><BrainCircuit size={15} /><span>INSTALLED TICK PROGRAM</span></div>
                  <code>{snapshot.artifacts.program[selectedArtifactIndex] || 'No autonomous program installed'}</code>
                  <small>Executed by the bounded deterministic VM on every simulation tick.</small>
                </div>
                <div className="program-card">
                  <div><Layers3 size={15} /><span>INVENTED GEOMETRY</span></div>
                  <div className="detail-pairs">
                    {Object.entries(snapshot.artifacts.geometry[selectedArtifactIndex] ?? {}).map(([name, value]) => (
                      <span key={name}><small>{name.replaceAll('_', ' ')}</small><strong>{Number(value).toFixed(3)}</strong></span>
                    ))}
                  </div>
                  {!Object.keys(snapshot.artifacts.geometry[selectedArtifactIndex] ?? {}).length && <small>No geometry parameters recorded.</small>}
                </div>
                <div className="program-card">
                  <div><Gauge size={15} /><span>MEASURED FIELD SERVICES</span></div>
                  <div className="service-table">
                    <span><b>Service</b><b>Current</b><b>Peak</b></span>
                    {Object.entries(snapshot.artifacts.services).map(([name, values]) => (
                      <span key={name}><small>{name.replaceAll('_', ' ')}</small><code>{Number(values[selectedArtifactIndex] ?? 0).toFixed(3)}</code><code>{Number(snapshot.artifacts.peak_services[name]?.[selectedArtifactIndex] ?? 0).toFixed(3)}</code></span>
                    ))}
                  </div>
                  <small>These values—not the invention's name—determine performance and novelty.</small>
                </div>
                {(snapshot.artifacts.causal_parents?.[selectedArtifactIndex]?.length ?? 0) > 0 && (
                  <div className="program-card">
                    <div><Network size={15} /><span>RECORDED CAUSAL PARENTS</span></div>
                    <div className="parent-id-list">{snapshot.artifacts.causal_parents![selectedArtifactIndex].map((parent) => <code key={parent}>{parent}</code>)}</div>
                  </div>
                )}
                <button className="wide-action" onClick={() => { setSelectedLineageNode(null); setTab('lineage') }}><Network size={14} /> Trace intellectual lineage</button>
              </div>
            )}

            {tab === 'inspect' && selectedTile && selectedTileIndex >= 0 && (
              <div className="inspector">
                <div className="selection-title tile-selection">
                  <div className="selection-icon"><MapPin size={18} /></div>
                  <div><small>SELECTED WORLD CELL</small><h2>{selectedTile[0]}, {selectedTile[1]}</h2><span>authoritative environmental state</span></div>
                </div>
                <div className="property-grid">
                  <div><span>Terrain</span><strong>{catalogName('terrains', snapshot.world.terrain[selectedTileIndex], TERRAIN_NAMES)}</strong></div>
                  <div><span>Station</span><strong>{catalogName('facilities', snapshot.world.stations[selectedTileIndex], STATION_NAMES)}</strong></div>
                  <div><span>Resource</span><strong>{catalogName('resources', snapshot.world.resource_kind[selectedTileIndex], RESOURCE_NAMES)}</strong></div>
                  <div><span>Resource mass</span><strong>{Number(snapshot.world.resource_mass[selectedTileIndex] ?? 0).toFixed(3)}</strong></div>
                  <div><span>Temperature</span><strong>{Number(snapshot.world.temperature[selectedTileIndex] ?? 0).toFixed(3)}</strong></div>
                  <div><span>Moisture</span><strong>{Number(snapshot.world.moisture[selectedTileIndex] ?? 0).toFixed(3)}</strong></div>
                  <div><span>Nutrients</span><strong>{Number(snapshot.world.nutrients[selectedTileIndex] ?? 0).toFixed(3)}</strong></div>
                  <div><span>Contamination</span><strong>{Number(snapshot.world.contamination[selectedTileIndex] ?? 0).toFixed(3)}</strong></div>
                  <div><span>Solar</span><strong>{Number(snapshot.world.solar[selectedTileIndex] ?? 0).toFixed(3)}</strong></div>
                  {scenario && Object.entries(snapshot.world.fields ?? {}).map(([name, values]) => (
                    <div key={name}><span>{name.replaceAll('_', ' ')}</span><strong>{Number(values[selectedTileIndex] ?? 0).toFixed(3)}</strong></div>
                  ))}
                </div>
                <div className="program-card">
                  <div><ScanSearch size={15} /><span>CELL OCCUPANTS</span></div>
                  <div className="world-object-list compact">
                    {selectedTileArtifacts.map(({ id, index }) => (
                      <button key={id} onClick={() => selectArtifact(id)}><Box size={13} /><span><strong>{snapshot.artifacts.name[index] || id}</strong><small>artifact · {id}</small></span></button>
                    ))}
                    {selectedTileAgents.map(({ id, index }) => (
                      <button key={id} onClick={() => selectAgent(id)}><CircleDot size={13} /><span><strong>{compactAgent(id)}</strong><small>{ACTION_NAMES[snapshot.agents.last_action[index]] ?? 'Unknown action'}</small></span></button>
                    ))}
                    {!selectedTileArtifacts.length && !selectedTileAgents.length && <small>No agents or artifacts currently occupy this cell.</small>}
                  </div>
                </div>
              </div>
            )}

            {tab === 'inspect' && selectedAgentIndex < 0 && selectedArtifactIndex < 0 && !selectedTile && (
              <div className="society-overview">
                <div className="empty-orbit"><div /><div /><FlaskConical size={31} /></div>
                <small>WORLD OVERVIEW</small>
                <h2>A decentralized biofabrication society</h2>
                <p>Select an agent or artifact in the world to inspect its current scientific state and intervene.</p>
                <div className="overview-stats">
                  <div><Gauge size={15} /><span>Mean energy</span><strong>{Math.round(averageEnergy * 100)}%</strong></div>
                  <div><BrainCircuit size={15} /><span>Insights</span><strong>{snapshot.metrics.deposited_insights}</strong></div>
                  <div><Hammer size={15} /><span>Artifacts</span><strong>{snapshot.artifacts.count}</strong></div>
                  <div><BookOpen size={15} /><span>Archive</span><strong>{snapshot.metrics.archive_entries}</strong></div>
                </div>
                {snapshot.research && (
                  <div className="research-card">
                    <div className="research-card-head"><FlaskConical size={15} /><span>COLLECTIVE SCIENCE SCORECARD</span><strong>{Math.round(snapshot.research.score * 100)}%</strong></div>
                    <div className="milestone-grid">
                      {Object.entries(snapshot.research.milestones).map(([name, reached]) => (
                        <div key={name} className={reached ? 'reached' : ''}><span>{reached ? '●' : '○'}</span>{name.replaceAll('_', ' ')}</div>
                      ))}
                    </div>
                    <small>
                      Outcome {snapshot.research.outcome_success ? 'passed' : 'pending'} · causal emergence {snapshot.research.emergent_success ? 'passed' : 'pending'} · composition synergy {(snapshot.research.composition_synergy ?? snapshot.research.composition_gain) === null ? 'not observed' : `${(snapshot.research.composition_synergy ?? snapshot.research.composition_gain)! >= 0 ? '+' : ''}${(snapshot.research.composition_synergy ?? snapshot.research.composition_gain)!.toFixed(3)}`}
                    </small>
                  </div>
                )}
                <div className="world-index">
                  <div className="world-index-head"><div><Search size={14} /><span>WORLD OBJECT INDEX</span></div><small>{snapshot.artifacts.count + snapshot.agents.count} objects</small></div>
                  <label className="object-search"><Search size={13} /><input value={objectQuery} onChange={(event) => setObjectQuery(event.target.value)} placeholder="Find artifact or agent…" /></label>
                  <div className="world-object-list">
                    {worldObjects.map((object) => (
                      <button key={`${object.kind}-${object.id}`} onClick={() => object.kind === 'artifact' ? selectArtifact(object.id) : selectAgent(object.id)}>
                        {object.kind === 'artifact' ? <Box size={13} /> : <CircleDot size={13} />}
                        <span><strong>{object.label}</strong><small>{object.kind} · {object.x}, {object.y} · {object.detail}</small></span>
                      </button>
                    ))}
                    {!worldObjects.length && <small>No world objects match this search.</small>}
                  </div>
                  {(snapshot.artifacts.count + snapshot.agents.count) > 80 && !objectQuery && <small className="index-note">Showing the first 80 objects. Search by identifier or name to locate another.</small>}
                </div>
                <button className="wide-action" onClick={() => setTab('report')}><Sparkles size={14} /> Summarize current situation</button>
              </div>
            )}

            {tab === 'activity' && (
              <div className="activity-view">
                <div className="section-head"><div><small>{playbackMode ? 'RECORDED EVENT STREAM' : 'LIVE EVENT STREAM'}</small><h2>Collective activity</h2></div><span>{displayedEvents.length} retained</span></div>
                <div className="action-mix" aria-label="Current action distribution">
                  <div><span>CURRENT ACTION MIX</span><small>{snapshot.agents.last_action.filter((action) => action > 0).length} active agents</small></div>
                  {actionLeaders.map(({ action, count }) => (
                    <div key={action} className="action-mix-row">
                      <span>{ACTION_NAMES[action] ?? `Action ${action}`}</span>
                      <i><b style={{ width: `${count / Math.max(1, snapshot.agents.count) * 100}%` }} /></i>
                      <strong>{count}</strong>
                    </div>
                  ))}
                  {!actionLeaders.length && <small className="no-action-mix">All agents are currently waiting.</small>}
                </div>
                <div className="activity-filters" aria-label="Filter activity events">
                  {(['all', 'messages', 'science', 'artifacts', 'errors'] as ActivityFilter[]).map((name) => (
                    <button key={name} className={activityFilter === name ? 'active' : ''} onClick={() => setActivityFilter(name)}>{name}</button>
                  ))}
                </div>
                <div className="event-list">
                  {[...filteredActivity].reverse().slice(0, 80).map((event, index) => (
                    <button key={`${event.tick}-${event.kind}-${index}`} className={`event-row ${eventCategory(event)}`} onClick={() => selectEventSubject(event)}>
                      <span className={`event-glyph ${event.kind}`}><Activity size={12} /></span>
                      <div>
                        <strong>{eventSummary(event)}</strong>
                        {eventDetail(event) && <p>{eventDetail(event)}</p>}
                        <small>{event.kind.replaceAll('_', ' ')}</small>
                      </div>
                      <time>T{event.tick}</time>
                    </button>
                  ))}
                  {!filteredActivity.length && <div className="no-events">No matching events in this window.</div>}
                </div>
              </div>
            )}

            {tab === 'lineage' && (
              <div className="lineage-view">
                <div className="section-head"><div><small>CAUSAL PROVENANCE</small><h2>Knowledge lineage</h2></div><Network size={18} /></div>
                <p>Typed arrows distinguish observation, testing, authorship, construction, program installation or forking, and causal dependence. Repeated recent messages are aggregated by directed agent pair.</p>
                <PanelErrorBoundary title="Knowledge lineage">
                  <Suspense fallback={<div className="network-view network-loading">Assembling provenance graph…</div>}>
                    <NetworkView
                      snapshot={snapshot}
                      events={displayedEvents}
                      selectedNodeId={selectedLineageNode?.id ?? selectedArtifact}
                      onSelectNode={setSelectedLineageNode}
                      onLayoutChange={setLineageGraphLayout}
                    />
                  </Suspense>
                </PanelErrorBoundary>
                {selectedLineageNode ? (
                  <div className={`lineage-node-detail ${selectedLineageNode.kind}`}>
                    <div className="node-detail-head"><span>{selectedLineageNode.kind.toUpperCase()}</span><code>{selectedLineageNode.id}</code></div>
                    <h3>{selectedLineageNode.label}</h3>
                    <div className="node-meta">
                      {selectedLineageNode.author && <span>Author <strong>{selectedLineageNode.author}</strong></span>}
                      {selectedLineageNode.tick !== undefined && <span>Tick <strong>{selectedLineageNode.tick}</strong></span>}
                      {selectedLineageNode.location && <span>Location <strong>{selectedLineageNode.location.join(', ')}</strong></span>}
                    </div>
                    {selectedLineageNode.content && <p>{selectedLineageNode.content}</p>}
                    {selectedLineageNode.properties && (
                      <div className="node-properties">
                        {Object.entries(selectedLineageNode.properties).map(([name, value]) => <span key={name}><small>{name}</small><strong>{value}</strong></span>)}
                      </div>
                    )}
                    {(selectedLineageNode.causalParents?.length ?? 0) > 0 && (
                      <div className="node-parents"><small>CAUSAL PARENTS</small>{selectedLineageNode.causalParents!.map((parent) => <code key={parent}>{parent}</code>)}</div>
                    )}
                    {selectedLineageNode.kind === 'artifact' && snapshot.artifacts.ids.includes(selectedLineageNode.id) && (
                      <button className="wide-action" onClick={() => selectArtifact(selectedLineageNode.id)}><Box size={13} /> Inspect artifact in world</button>
                    )}
                    {selectedLineageNode.kind === 'agent' && snapshot.agents.ids.includes(selectedLineageNode.agentId ?? selectedLineageNode.id) && (
                      <button className="wide-action" onClick={() => selectAgent(selectedLineageNode.agentId ?? selectedLineageNode.id)}><CircleDot size={13} /> Inspect agent in world</button>
                    )}
                  </div>
                ) : <div className="lineage-node-hint"><ScanSearch size={13} /> Select any node to read its recorded contents and provenance.</div>}
                <div className="network-legend"><span><i className="insight" />Insight</span><span><i className="evidence" />Evidence</span><span><i className="artifact" />Artifact</span><span><i className="agent" />Agent</span><span><i className="program" />Program</span></div>
                <div className="network-relation-legend"><span><i className="observed" />observed</span><span><i className="tested" />tested</span><span><i className="built" />built</span><span><i className="program-edge" />installed / forked</span><span><i className="communicated" />messages ×n</span></div>
              </div>
            )}

            {tab === 'report' && <SituationReport snapshot={snapshot} events={displayedEvents} />}
          </div>
        </aside>

        <section className={`analytics-dock glass-panel ${analyticsOpen ? 'open' : 'closed'}`}>
          <button className="dock-toggle" onClick={() => setAnalyticsOpen((value) => !value)} aria-label={analyticsOpen ? 'Collapse analytics' : 'Expand analytics'}>
            <ChevronDown size={14} />
          </button>
          <div className="dock-title"><Activity size={14} /><div><strong>Society dynamics</strong><span>{playbackMode ? 'Complete recorded metrics · exact tick playback' : 'Full-run metrics · recoverable sampled world replay'}</span></div></div>
          {analyticsOpen && <MetricChart history={dynamicsHistory} />}
          <div className="timeline">
            <button className={replayIndex < 0 ? 'live' : ''} onClick={() => setReplayIndex(-1)}><Radio size={12} />{playbackMode ? 'PLAYHEAD' : 'LIVE'}</button>
            <button
              aria-label="Replay from beginning"
              title="Replay from beginning"
              disabled={history.length < 2}
              onClick={() => playbackMode ? sendCommand({ command: 'seek', tick: 0 }) : setReplayIndex(0)}
            ><RotateCcw size={12} />FIRST</button>
            <input
              type="range"
              min={0}
              max={Math.max(0, history.length - 1)}
              value={timelineValue}
              disabled={history.length < 2}
              onInput={(event) => setReplayIndex(Number(event.currentTarget.value))}
              onChange={(event) => seekToHistoryIndex(Number(event.currentTarget.value))}
              aria-label="Replay timeline"
            />
            <span>{replayIndex >= 0 ? `T${history[timelineValue]?.snapshot.tick ?? snapshot.tick}` : `T${snapshot.tick}`}</span>
          </div>
        </section>

        <div className="protocol-note"><Info size={12} /> Protocol v{snapshot.protocol} · seed {snapshot.seed} · authoritative state remains in Python</div>
      </main>
    </div>
  )
}

export default App
