import { EVENT_LABELS, FIELD_OVERLAYS, RESOURCE_NAMES } from './constants.js'

function el(id) {
  return document.getElementById(id)
}

function affordanceTruthy(affordances, ...needles) {
  if (!affordances || typeof affordances !== 'object') return false
  for (const [key, value] of Object.entries(affordances)) {
    const lower = key.toLowerCase()
    if (needles.some((needle) => lower.includes(needle))) {
      if (Array.isArray(value)) return value.length > 0
      if (typeof value === 'object' && value !== null) return Object.keys(value).length > 0
      if (value) return true
    }
  }
  return false
}

export function modeLabel(settings) {
  if (!settings) return '—'
  if (settings.require_response) return 'TURN-BASED'
  if (settings.input_timeout_seconds > 0) return `PROMPTED ${settings.input_timeout_seconds}s`
  return 'REAL-TIME'
}

export function updateHud(state) {
  const control = state.control || {}
  el('conn-dot').classList.toggle('on', state.connected)
  el('status-tick').textContent = `tick ${state.snapshot ? state.snapshot.tick : '—'}`
  el('status-mode').textContent = modeLabel(state.game?.settings)
  el('status-speed').textContent = control.paused
    ? 'paused'
    : `×${control.speed_multiplier ?? 1}`
  el('status-model').textContent = control.llm_enabled
    ? `${control.model ?? 'llm'} · ${control.model_requests ?? 0} calls` +
      (control.model_errors ? ` · ${control.model_errors} errors` : '')
    : 'scripted society'
  const banner = []
  if (control.provider_outage) banner.push('provider outage — retrying')
  if (state.completed) banner.push('society reached its final tick')
  if (state.thinkingBanner) banner.push('agents thinking…')
  el('banner').textContent = banner.join(' · ')

  const overlay = FIELD_OVERLAYS[state.fieldOverlay]
  const joinButton = el('join-btn')
  const game = state.game || {}
  if (state.possessed) {
    joinButton.textContent = `Release ${game.human_agent}`
    joinButton.classList.add('joined')
    joinButton.disabled = false
  } else if (game.joined) {
    joinButton.textContent = 'Slot taken by another player'
    joinButton.classList.remove('joined')
    joinButton.disabled = true
  } else {
    joinButton.textContent = `Join as ${game.human_agent ?? '…'}`
    joinButton.classList.remove('joined')
    joinButton.disabled = !state.connected
  }
  el('join-info').textContent = state.possessed
    ? `you are embodied · ${overlay.label}`
    : `spectating · ${overlay.label}`

  const observation = state.humanObs
  const self = observation?.self
  el('player-card').style.opacity = state.possessed && observation ? 1 : 0.45
  if (self) {
    const energy = Math.max(0, Math.min(1, Number(self.energy ?? 1)))
    el('energy-fill').style.width = `${Math.round(energy * 100)}%`
    const inventory = self.inventory && typeof self.inventory === 'object' ? self.inventory : {}
    const entries = Object.entries(inventory)
    el('inventory').innerHTML =
      entries.length === 0
        ? '<div class="k">empty</div>'
        : entries
            .map(
              ([name, mass]) =>
                `<div><span class="k">${name}</span><span>${Number(mass).toFixed(2)}</span></div>`,
            )
            .join('')
  }
  const local = observation?.local
  if (local) {
    const rows = []
    for (const key of [
      'terrain',
      'station',
      'resource',
      'resource_mass',
      'moisture',
      'nutrients',
      'temperature',
      'contamination',
      'solar',
    ]) {
      if (local[key] === undefined || local[key] === null) continue
      let value = local[key]
      if (typeof value === 'number') value = value.toFixed(2)
      rows.push(`<div><span class="k">${key.replace('_', ' ')}</span><span>${value}</span></div>`)
    }
    el('tile-info').innerHTML = rows.join('')
  }

  const affordances = observation?.local_affordances
  el('affordance-note').textContent = state.possessed && !affordances ? '(no observation yet)' : ''
  const canAct = state.possessed && state.connected
  for (const button of document.querySelectorAll('#dpad button, #verbs button')) {
    button.disabled = !canAct
  }
  if (canAct && affordances) {
    // Exact engine affordance keys (simulation.semantic_observation), with a
    // fuzzy fallback for scenario packages that may extend them.
    const gate = (selector, exactKey, ...needles) => {
      const button = document.querySelector(selector)
      if (!button) return
      const exact = affordances[exactKey]
      const truthy =
        exact !== undefined
          ? Array.isArray(exact)
            ? exact.length > 0
            : Boolean(exact)
          : affordanceTruthy(affordances, ...needles)
      button.disabled = !truthy
    }
    gate('[data-act="HARVEST"]', 'collectable_material_here', 'collect', 'harvest')
    gate('[data-act="DEPOSIT"]', 'shared_depot_access_here', 'depot', 'deposit')
    gate('[data-act="TEST"]', 'untested_microbatch_ready', 'test', 'microbatch')
    const walkable = affordances.walkable_directions
    if (Array.isArray(walkable)) {
      for (const button of document.querySelectorAll('#dpad button[data-dir]')) {
        button.disabled = !walkable.includes(button.dataset.dir)
      }
    }
  }
}

export function renderChat(state) {
  const log = el('chat-log')
  log.innerHTML = state.chat
    .slice(-60)
    .map(
      (entry) =>
        `<div class="msg${entry.mine ? ' mine' : ''}">` +
        `<span class="who">${entry.who}</span> <span class="t">t${entry.tick}</span> ` +
        `${entry.text}</div>`,
    )
    .join('')
  log.scrollTop = log.scrollHeight
}

export function describeEvent(event, me) {
  const label = EVENT_LABELS[event.kind]
  if (label === null) return null
  const payload = event.payload || {}
  const actor = payload.agent || payload.sender || payload.creator || ''
  let detail = ''
  if (event.kind === 'message_delivered') {
    detail = `“${String(payload.message ?? '').slice(0, 60)}”`
  } else if (event.kind === 'action_rejected') {
    detail = String(payload.reason ?? payload.detail ?? '').slice(0, 70)
  } else if (event.kind === 'resource_harvested') {
    detail = `${RESOURCE_NAMES[payload.resource] ?? ''} ${payload.amount ?? ''}`
  } else if (event.kind === 'artifact_built' || event.kind === 'artifact_milestone') {
    detail = String(payload.name ?? payload.artifact ?? '').slice(0, 40)
  } else if (event.kind === 'environmental_disturbance') {
    detail = String(payload.kind ?? payload.disturbance ?? '')
  }
  const involvesMe =
    me &&
    (actor === me ||
      payload.recipient === me ||
      (Array.isArray(payload.recipients) && payload.recipients.includes(me)))
  return {
    tick: event.tick,
    label: label ?? event.kind,
    actor,
    detail,
    hot: Boolean(involvesMe),
    reject: event.kind === 'action_rejected' && involvesMe,
  }
}

export function renderEvents(state) {
  const me = state.game?.human_agent
  const rows = []
  for (let i = state.events.length - 1; i >= 0 && rows.length < 40; i -= 1) {
    const described = describeEvent(state.events[i], state.possessed ? me : null)
    if (!described) continue
    rows.push(
      `<div class="evt${described.hot ? ' hot' : ''}${described.reject ? ' reject' : ''}">` +
        `<span class="t">t${described.tick}</span> ${described.label}` +
        `${described.actor ? ` · ${described.actor.replace('agent_00000', 'a')}` : ''}` +
        `${described.detail ? ` · ${described.detail}` : ''}</div>`,
    )
  }
  el('events-log').innerHTML = rows.join('')
}

export function toast(text, isError = false) {
  const container = el('toasts')
  const node = document.createElement('div')
  node.className = `toast${isError ? ' err' : ''}`
  node.textContent = text
  container.appendChild(node)
  setTimeout(() => node.remove(), 4200)
  while (container.children.length > 5) container.firstChild.remove()
}

export function updateDecision(state, now) {
  const box = el('decision')
  const decision = state.decision
  if (!decision || !state.possessed) {
    box.classList.add('hidden')
    return
  }
  box.classList.remove('hidden')
  if (decision.deadline === null) {
    el('decision-label').textContent = 'your move — the world is waiting'
    el('decision-fill').style.width = '100%'
  } else {
    const remaining = Math.max(0, decision.deadline - now)
    if (remaining <= 0) {
      box.classList.add('hidden')
      state.decision = null
      return
    }
    el('decision-label').textContent = 'your move'
    el('decision-fill').style.width = `${(remaining / decision.total) * 100}%`
  }
}
