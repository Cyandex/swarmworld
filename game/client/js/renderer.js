import {
  AGENT_COLORS,
  FIELD_OVERLAYS,
  RESOURCE_COLORS,
  STATION_NAMES,
  TERRAIN_COLORS,
} from './constants.js'

const TAU = Math.PI * 2

export class WorldRenderer {
  constructor(canvas) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    this.camera = { x: 24, y: 18, zoom: 20 }
    this.resize()
  }

  resize() {
    const ratio = window.devicePixelRatio || 1
    const { clientWidth, clientHeight } = this.canvas
    this.canvas.width = Math.max(1, Math.floor(clientWidth * ratio))
    this.canvas.height = Math.max(1, Math.floor(clientHeight * ratio))
    this.ratio = ratio
  }

  screenToTile(px, py, world) {
    const cell = this.camera.zoom
    const w = this.canvas.clientWidth
    const h = this.canvas.clientHeight
    const tx = Math.floor((px - w / 2) / cell + this.camera.x)
    const ty = Math.floor((py - h / 2) / cell + this.camera.y)
    if (!world || tx < 0 || ty < 0 || tx >= world.width || ty >= world.height) return null
    return { x: tx, y: ty }
  }

  draw(state, now) {
    const ctx = this.ctx
    const width = this.canvas.clientWidth
    const height = this.canvas.clientHeight
    ctx.setTransform(this.ratio, 0, 0, this.ratio, 0, 0)
    ctx.fillStyle = '#0d1117'
    ctx.fillRect(0, 0, width, height)
    const snapshot = state.snapshot
    if (!snapshot) {
      ctx.fillStyle = '#8395ab'
      ctx.font = '14px ui-monospace, monospace'
      ctx.textAlign = 'center'
      ctx.fillText(state.connected ? 'waiting for world…' : 'connecting…', width / 2, height / 2)
      return
    }
    const world = snapshot.world
    const cell = this.camera.zoom
    const toX = (tx) => (tx - this.camera.x) * cell + width / 2
    const toY = (ty) => (ty - this.camera.y) * cell + height / 2

    const x0 = Math.max(0, Math.floor(this.camera.x - width / 2 / cell) - 1)
    const x1 = Math.min(world.width - 1, Math.ceil(this.camera.x + width / 2 / cell) + 1)
    const y0 = Math.max(0, Math.floor(this.camera.y - height / 2 / cell) - 1)
    const y1 = Math.min(world.height - 1, Math.ceil(this.camera.y + height / 2 / cell) + 1)

    const overlay = FIELD_OVERLAYS[state.fieldOverlay]
    const field = overlay?.key ? world[overlay.key] : null

    for (let ty = y0; ty <= y1; ty += 1) {
      for (let tx = x0; tx <= x1; tx += 1) {
        const index = ty * world.width + tx
        ctx.fillStyle = TERRAIN_COLORS[world.terrain[index]] || '#222'
        ctx.fillRect(toX(tx), toY(ty), cell + 0.5, cell + 0.5)
        if (field) {
          const value = Math.max(0, Math.min(1, field[index]))
          const [r, g, b] = overlay.color
          ctx.fillStyle = `rgba(${r},${g},${b},${value * 0.55})`
          ctx.fillRect(toX(tx), toY(ty), cell + 0.5, cell + 0.5)
        }
      }
    }

    if (cell >= 14) {
      ctx.strokeStyle = 'rgba(0,0,0,0.18)'
      ctx.lineWidth = 1
      ctx.beginPath()
      for (let tx = x0; tx <= x1 + 1; tx += 1) {
        ctx.moveTo(toX(tx), toY(y0))
        ctx.lineTo(toX(tx), toY(y1 + 1))
      }
      for (let ty = y0; ty <= y1 + 1; ty += 1) {
        ctx.moveTo(toX(x0), toY(ty))
        ctx.lineTo(toX(x1 + 1), toY(ty))
      }
      ctx.stroke()
    }

    // Resource deposits
    for (let ty = y0; ty <= y1; ty += 1) {
      for (let tx = x0; tx <= x1; tx += 1) {
        const index = ty * world.width + tx
        const kind = world.resource_kind[index]
        if (!kind) continue
        const mass = Math.max(0, Math.min(1, world.resource_mass[index]))
        if (mass <= 0.01) continue
        const radius = cell * (0.10 + 0.24 * Math.sqrt(mass))
        ctx.fillStyle = RESOURCE_COLORS[kind] || '#fff'
        ctx.globalAlpha = 0.85
        ctx.beginPath()
        ctx.arc(toX(tx) + cell / 2, toY(ty) + cell / 2, radius, 0, TAU)
        ctx.fill()
        ctx.globalAlpha = 1
      }
    }

    // Stations
    for (let ty = y0; ty <= y1; ty += 1) {
      for (let tx = x0; tx <= x1; tx += 1) {
        const index = ty * world.width + tx
        const station = world.stations[index]
        if (!station) continue
        const pad = cell * 0.18
        ctx.strokeStyle = '#dfe7f2'
        ctx.lineWidth = Math.max(1, cell * 0.08)
        ctx.strokeRect(toX(tx) + pad, toY(ty) + pad, cell - pad * 2, cell - pad * 2)
        if (cell >= 16) {
          ctx.fillStyle = '#dfe7f2'
          ctx.font = `${Math.floor(cell * 0.42)}px ui-monospace, monospace`
          ctx.textAlign = 'center'
          ctx.textBaseline = 'middle'
          ctx.fillText(
            (STATION_NAMES[station] || '?')[0],
            toX(tx) + cell / 2,
            toY(ty) + cell / 2 + 1,
          )
        }
      }
    }

    // Artifacts: diamonds colored by health
    const artifacts = snapshot.artifacts || {}
    const artifactCount = (artifacts.ids || []).length
    for (let i = 0; i < artifactCount; i += 1) {
      if (artifacts.retired?.[i]) continue
      const ax = toX(artifacts.x[i]) + cell / 2
      const ay = toY(artifacts.y[i]) + cell / 2
      if (ax < -cell || ay < -cell || ax > width + cell || ay > height + cell) continue
      const health = Math.max(0, Math.min(1, artifacts.health?.[i] ?? 1))
      const size = cell * 0.34
      ctx.fillStyle = `hsl(${Math.round(10 + 130 * health)}, 65%, 55%)`
      ctx.strokeStyle = '#0d1117'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(ax, ay - size)
      ctx.lineTo(ax + size, ay)
      ctx.lineTo(ax, ay + size)
      ctx.lineTo(ax - size, ay)
      ctx.closePath()
      ctx.fill()
      ctx.stroke()
      if (cell >= 22 && artifacts.name?.[i]) {
        ctx.fillStyle = 'rgba(220,230,242,0.8)'
        ctx.font = '10px ui-monospace, monospace'
        ctx.textAlign = 'center'
        ctx.fillText(String(artifacts.name[i]).slice(0, 18), ax, ay - size - 4)
      }
    }

    // Agents, interpolated between the two latest authoritative frames
    const agents = snapshot.agents
    const prev = state.prevSnapshot?.agents
    const span = Math.max(40, state.snapshotAt - state.prevAt)
    const t = Math.max(0, Math.min(1, (now - state.snapshotAt) / span))
    const positions = new Map()
    for (let i = 0; i < agents.ids.length; i += 1) {
      if (agents.active && agents.active[i] === false) continue
      let ax = agents.x[i]
      let ay = agents.y[i]
      if (prev) {
        const j = prev.ids.indexOf(agents.ids[i])
        if (j >= 0) {
          ax = prev.x[j] + (agents.x[i] - prev.x[j]) * t
          ay = prev.y[j] + (agents.y[i] - prev.y[j]) * t
        }
      }
      positions.set(agents.ids[i], { x: ax, y: ay, index: i })
    }

    const me = state.game?.human_agent
    const mine = state.possessed ? positions.get(me) : null

    // Player perception envelope: sensing square (Chebyshev 3) + comm radius
    if (mine) {
      const px = toX(mine.x) + cell / 2
      const py = toY(mine.y) + cell / 2
      ctx.strokeStyle = 'rgba(104,213,255,0.25)'
      ctx.setLineDash([4, 4])
      ctx.lineWidth = 1
      ctx.strokeRect(toX(mine.x - 3), toY(mine.y - 3), cell * 7, cell * 7)
      ctx.beginPath()
      ctx.arc(px, py, cell * 6, 0, TAU)
      ctx.stroke()
      ctx.setLineDash([])
    }

    for (const [agentId, pos] of positions) {
      const px = toX(pos.x) + cell / 2
      const py = toY(pos.y) + cell / 2
      if (px < -cell || py < -cell || px > width + cell || py > height + cell) continue
      const isMe = agentId === me && state.possessed
      const visor = agents.visor ? agents.visor[pos.index] : pos.index
      const radius = cell * (isMe ? 0.34 : 0.28)
      if (isMe) {
        ctx.fillStyle = 'rgba(255,255,255,0.15)'
        ctx.beginPath()
        ctx.arc(px, py, radius * 1.8, 0, TAU)
        ctx.fill()
      }
      ctx.fillStyle = AGENT_COLORS[visor % AGENT_COLORS.length]
      ctx.strokeStyle = isMe ? '#ffffff' : '#0d1117'
      ctx.lineWidth = isMe ? 2 : 1.2
      ctx.beginPath()
      ctx.arc(px, py, radius, 0, TAU)
      ctx.fill()
      ctx.stroke()
      const thinkingUntil = state.thinking.get(agentId)
      if (thinkingUntil && thinkingUntil > now) {
        ctx.fillStyle = 'rgba(220,230,242,0.9)'
        ctx.font = `${Math.max(10, cell * 0.5)}px ui-monospace, monospace`
        ctx.textAlign = 'center'
        ctx.fillText('…', px, py - radius - 3)
      }
      if (cell >= 24 || isMe) {
        ctx.fillStyle = isMe ? '#ffffff' : 'rgba(220,230,242,0.65)'
        ctx.font = '9px ui-monospace, monospace'
        ctx.textAlign = 'center'
        ctx.fillText(agentId.replace('agent_00000', 'a').replace('agent_0000', 'a'), px, py + radius + 10)
      }
    }

    // Selected tile highlight
    if (state.selected) {
      ctx.strokeStyle = '#ffc857'
      ctx.lineWidth = 2
      ctx.strokeRect(toX(state.selected.x) + 1, toY(state.selected.y) + 1, cell - 2, cell - 2)
    }
  }
}
