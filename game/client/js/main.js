import { GameSocket } from './socket.js'
import { WorldRenderer } from './renderer.js'
import { bindInput } from './input.js'
import { renderChat, renderEvents, toast, updateDecision, updateHud } from './hud.js'

const canvas = document.getElementById('world')
const renderer = new WorldRenderer(canvas)

const state = {
  connected: false,
  snapshot: null,
  prevSnapshot: null,
  snapshotAt: 0,
  prevAt: 0,
  control: null,
  game: null,
  humanObs: null,
  possessed: false,
  decision: null,
  events: [],
  chat: [],
  thinking: new Map(),
  thinkingBanner: false,
  fieldOverlay: 0,
  follow: true,
  selected: null,
  completed: false,
  lastFrameAt: 0,
}

let requestCounter = 0
let hudDirty = true

function acceptSnapshot(snapshot) {
  state.prevSnapshot = state.snapshot
  state.prevAt = state.snapshotAt
  state.snapshot = snapshot
  state.snapshotAt = performance.now()
  state.lastFrameAt = state.snapshotAt
  if (state.prevSnapshot === null && snapshot?.world) {
    renderer.camera.x = snapshot.world.width / 2
    renderer.camera.y = snapshot.world.height / 2
  }
}

function ingestEvents(events, tickNow) {
  const me = state.game?.human_agent
  for (const event of events || []) {
    state.events.push(event)
    if (event.kind === 'agent_deliberated' || event.kind === 'model_error') {
      state.thinking.set(event.payload?.agent, performance.now() + 2200)
    }
    if (event.kind === 'message_delivered') {
      const payload = event.payload || {}
      const isMine = payload.sender === me && state.possessed
      const heardByMe =
        !me ||
        !state.possessed ||
        isMine ||
        (Array.isArray(payload.recipients) && payload.recipients.includes(me))
      if (heardByMe) {
        state.chat.push({
          tick: event.tick ?? tickNow,
          who: (payload.sender ?? '?').replace('agent_00000', 'a'),
          text: String(payload.message ?? ''),
          mine: isMine,
        })
      }
    }
    if (event.kind === 'human_intervention' && state.possessed) {
      state.thinking.delete(me)
    }
    if (event.kind === 'action_rejected' && state.possessed && event.payload?.agent === me) {
      toast(`rejected: ${String(event.payload.reason ?? '').slice(0, 90)}`, true)
    }
    if (event.kind === 'environmental_disturbance') {
      toast(`⚠ disturbance: ${event.payload?.kind ?? ''}`, true)
    }
  }
  if (state.events.length > 600) state.events = state.events.slice(-400)
  if (state.chat.length > 120) state.chat = state.chat.slice(-80)
}

function handleMessage(message) {
  switch (message.type) {
    case 'snapshot':
      acceptSnapshot(message.snapshot)
      state.control = message.control ?? state.control
      if (message.game) state.game = message.game
      break
    case 'frame':
      acceptSnapshot(message.snapshot)
      state.control = message.control ?? state.control
      if (message.game) state.game = message.game
      if (message.human_observation !== undefined) state.humanObs = message.human_observation
      state.decision = null
      ingestEvents(message.events, message.snapshot?.tick)
      break
    case 'complete':
      acceptSnapshot(message.snapshot)
      state.control = message.control ?? state.control
      state.completed = true
      break
    case 'heartbeat':
      state.control = message.control ?? state.control
      break
    case 'control':
      state.control = message.control ?? state.control
      if (message.detail) toast(message.detail, true)
      break
    case 'game_state': {
      state.game = message
      if (!message.joined && state.possessed) {
        state.possessed = false
        toast('possession released', false)
      }
      break
    }
    case 'join_ack':
      if (message.accepted) {
        state.possessed = true
        state.game = message.game ?? state.game
        state.follow = true
        toast(`you are ${message.agent}`, false)
      } else {
        toast(message.detail || 'join refused', true)
      }
      break
    case 'release_ack':
      if (message.accepted) state.possessed = false
      state.game = message.game ?? state.game
      break
    case 'action_ack':
      if (!message.accepted) toast(`not accepted: ${message.detail}`, true)
      break
    case 'decision_prompt': {
      const total = message.deadline_ms
      state.decision = {
        tick: message.tick,
        deadline: total === null ? null : performance.now() + total,
        total: total ?? 0,
        affordances: message.affordances,
      }
      break
    }
    case 'provider_outage':
      state.control = message.control ?? state.control
      break
    default:
      break // replay_frame / replay_ready / dynamics packets are Observatory features
  }
  hudDirty = true
}

const socketUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`
const socket = new GameSocket(socketUrl, {
  onMessage: handleMessage,
  onStatus: (connected) => {
    state.connected = connected
    if (!connected && state.possessed) state.possessed = false
    hudDirty = true
  },
})
window.__gameDebug = { state, socket }

function send(payload) {
  if (!socket.send(payload)) toast('not connected', true)
}

function act(action) {
  if (!state.possessed) {
    toast('join first (button on the right)', true)
    return
  }
  requestCounter += 1
  send({ command: 'human_action', request_id: `r${requestCounter}`, action })
}

bindInput({ canvas, state, renderer, send, act })

window.addEventListener('resize', () => renderer.resize())

function frame(now) {
  // Camera follow with smoothing
  if (state.follow && state.possessed && state.snapshot) {
    const me = state.game?.human_agent
    const agents = state.snapshot.agents
    const index = agents.ids.indexOf(me)
    if (index >= 0) {
      const k = 1 - Math.exp(-(now - (frame.last ?? now)) / 180)
      renderer.camera.x += (agents.x[index] + 0.5 - renderer.camera.x) * k
      renderer.camera.y += (agents.y[index] + 0.5 - renderer.camera.y) * k
    }
  }
  frame.last = now

  // "Society thinking" banner: LLM mode with a stalled tick stream
  const llm = state.control?.llm_enabled
  const stalled = state.lastFrameAt && now - state.lastFrameAt > 2600
  const thinkingBanner = Boolean(llm && stalled && !state.control?.paused && !state.completed)
  if (thinkingBanner !== state.thinkingBanner) {
    state.thinkingBanner = thinkingBanner
    hudDirty = true
  }

  renderer.draw(state, now)
  updateDecision(state, now)
  if (hudDirty) {
    hudDirty = false
    updateHud(state)
    renderChat(state)
    renderEvents(state)
  }
  requestAnimationFrame(frame)
}

requestAnimationFrame(frame)
