import type { ControlState, Snapshot, WorldEvent } from '../types'

function mulberry32(seed: number) {
  let value = seed >>> 0
  return () => {
    value += 0x6d2b79f5
    let result = value
    result = Math.imul(result ^ (result >>> 15), result | 1)
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61)
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296
  }
}

export const DEMO_CONTROL: ControlState = {
  paused: false,
  speed_multiplier: 1,
  step_budget: 0,
  model_requests: 84,
  model_errors: 1,
  provider_attempts: 84,
  provider_outage: false,
  llm_enabled: true,
  model: 'demonstration society',
}

export function createDemoSnapshot(seed = 17, tick = 156): Snapshot {
  const random = mulberry32(seed)
  const width = 48
  const height = 36
  const size = width * height
  const terrain = new Array<number>(size).fill(2)
  const resourceKind = new Array<number>(size).fill(0)
  const resourceMass = new Array<number>(size).fill(0)
  const stations = new Array<number>(size).fill(0)
  const temperature = new Array<number>(size).fill(0)
  const moisture = new Array<number>(size).fill(0)
  const nutrients = new Array<number>(size).fill(0)
  const contamination = new Array<number>(size).fill(0)
  const solar = new Array<number>(size).fill(0)

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const i = y * width + x
      const nx = x / (width - 1)
      const ny = y / (height - 1)
      const coast = 0.12 + 0.035 * Math.sin(ny * Math.PI * 5)
      const fungal = ((nx - 0.81) / 0.22) ** 2 + ((ny - 0.22) / 0.28) ** 2 < 1
      const chitin = ((nx - 0.25) / 0.22) ** 2 + ((ny - 0.78) / 0.25) ** 2 < 1
      const cellulose = ((nx - 0.33) / 0.2) ** 2 + ((ny - 0.42) / 0.22) ** 2 < 1
      const mineral = ((nx - 0.82) / 0.14) ** 2 + ((ny - 0.73) / 0.17) ** 2 < 1

      if (nx < coast * 0.58) terrain[i] = 0
      else if (nx < coast) terrain[i] = 1
      if (fungal) terrain[i] = 3
      if (chitin) terrain[i] = 4
      if (cellulose) terrain[i] = 5
      if (mineral) terrain[i] = 6
      if (x >= 21 && x <= 33 && y >= 14 && y <= 24) terrain[i] = 7
      if (x >= 23 && x <= 35 && y >= 26 && y <= 32) terrain[i] = 8

      if (terrain[i] === 1) resourceKind[i] = (x + y * 2) % 3 === 0 ? 2 : 1
      if (terrain[i] === 3) resourceKind[i] = 3
      if (terrain[i] === 4) resourceKind[i] = 4
      if (terrain[i] === 5) resourceKind[i] = 5
      if (terrain[i] === 6) resourceKind[i] = 6
      resourceMass[i] = resourceKind[i] ? 1.6 + random() * 1.4 : 0
      const noise = (random() - 0.5) * 0.05
      moisture[i] = Math.max(0, Math.min(1, 0.35 + 0.45 * (1 - nx) + noise + (fungal ? 0.15 : 0)))
      nutrients[i] = Math.max(0, Math.min(1, 0.25 + (fungal ? 0.45 : 0) + (chitin ? 0.25 : 0) + noise))
      contamination[i] = Math.max(0, random() * 0.025 + Math.exp(-((x - 30) ** 2 + (y - 20) ** 2) / 28) * 0.19)
      solar[i] = Math.max(0, Math.min(1, 0.75 - (fungal ? 0.3 : 0) + 0.12 * Math.sin(nx * Math.PI)))
      temperature[i] = Math.max(0, Math.min(1, 0.62 - 0.2 * ny + noise))
    }
  }

  ;[
    [23, 16, 2], [27, 16, 1], [31, 16, 4], [23, 21, 3], [27, 21, 5], [31, 21, 6],
  ].forEach(([x, y, kind]) => { stations[y * width + x] = kind })

  const count = 42
  const ids = Array.from({ length: count }, (_, i) => `agent_${i.toString().padStart(6, '0')}`)
  const x = Array.from({ length: count }, (_, i) => 8 + ((i * 7 + Math.floor(random() * 4)) % 34))
  const y = Array.from({ length: count }, (_, i) => 5 + ((i * 11 + Math.floor(random() * 3)) % 27))
  const artifactCount = 9

  return {
    protocol: 1,
    tick,
    seed,
    world: {
      width, height, terrain, resource_kind: resourceKind, resource_mass: resourceMass,
      stations, temperature, moisture, nutrients, contamination, solar,
    },
    agents: {
      count,
      display_count: count,
      ids,
      x,
      y,
      energy: Array.from({ length: count }, () => 0.55 + random() * 0.44),
      visor: Array.from({ length: count }, (_, i) => i % 12),
      inventory_total: Array.from({ length: count }, () => random() * 8.5),
      last_action: Array.from({ length: count }, (_, i) => [1, 1, 2, 3, 5, 7, 11, 13, 14][i % 9]),
    },
    artifacts: {
      count: artifactCount,
      display_count: artifactCount,
      ids: Array.from({ length: artifactCount }, (_, i) => `artifact_${i.toString().padStart(8, '0')}`),
      kind: Array.from({ length: artifactCount }, () => 1),
      x: [19, 25, 29, 34, 15, 39, 27, 32, 22],
      y: [18, 13, 23, 17, 27, 8, 29, 27, 7],
      health: Array.from({ length: artifactCount }, () => 0.68 + random() * 0.3),
      maturity: Array.from({ length: artifactCount }, () => 0.35 + random() * 0.63),
      performance: Array.from({ length: artifactCount }, () => 0.25 + random() * 0.65),
      peak_performance: Array.from({ length: artifactCount }, () => 0.55 + random() * 0.42),
      storage: Array.from({ length: artifactCount }, () => random()),
      open_fraction: Array.from({ length: artifactCount }, () => 0.2 + random() * 0.75),
      program: ['wicking_loop', '', 'solar_response', 'repair_cycle', 'bridge_growth', '', 'shade_control', '', 'load_balance'],
      name: ['Lamellar catchment', 'Branching stress web', 'Solar pore skin', 'Crack-seeking film', 'Rootlike brace', 'Mineral moisture veil', 'Circadian aperture mesh', 'Adhesive detox layer', 'Graded load lattice'],
      claimed_function: Array.from({ length: artifactCount }, (_, i) => `Demonstration hypothesis ${i + 1} for a locally adaptive habitat service`),
      architecture: Array.from({ length: artifactCount }, (_, i) => `Agent-authored continuous architecture variant ${i + 1}`),
      bio_inspiration: Array.from({ length: artifactCount }, (_, i) => [i % 2 ? 'plant vasculature' : 'marine interfaces']),
      geometry: Array.from({ length: artifactCount }, (_, i) => ({ layers: 1 + (i % 5), surface_area: 0.8 + random() * 1.7, channel_density: random(), anisotropy: random(), branching: random(), connectivity: random(), curvature: random(), modularity: random() })),
      services: {
        water_capture: Array.from({ length: artifactCount }, () => random() * 0.8),
        remediation: Array.from({ length: artifactCount }, () => random() * 0.5),
        structural_support: Array.from({ length: artifactCount }, () => random() * 0.85),
        adaptive_regulation: Array.from({ length: artifactCount }, () => random() * 0.65),
        self_maintenance: Array.from({ length: artifactCount }, () => random() * 0.45),
        ecological_support: Array.from({ length: artifactCount }, () => random() * 0.55),
      },
      peak_services: {
        water_capture: Array.from({ length: artifactCount }, () => 0.25 + random() * 0.7),
        remediation: Array.from({ length: artifactCount }, () => 0.2 + random() * 0.65),
        structural_support: Array.from({ length: artifactCount }, () => 0.3 + random() * 0.65),
        adaptive_regulation: Array.from({ length: artifactCount }, () => 0.2 + random() * 0.7),
        self_maintenance: Array.from({ length: artifactCount }, () => 0.1 + random() * 0.7),
        ecological_support: Array.from({ length: artifactCount }, () => 0.15 + random() * 0.7),
      },
    },
    archive: [
      { id: 'record_104', author: 'agent_000018', title: 'Humidity-responsive lamella geometry', tick: tick - 28 },
      { id: 'record_109', author: 'agent_000031', title: 'Mycelium bridge load redistribution', tick: tick - 17 },
      { id: 'record_112', author: 'agent_000007', title: 'Mineral shell crack arrest', tick: tick - 6 },
    ],
    insights: [
      { id: 'insight_41', author: 'agent_000018', title: 'Alternating wet–dry cycles increase membrane transport', x: 25, y: 14, tick: tick - 31, causal_parents: [] },
      { id: 'insight_47', author: 'agent_000031', title: 'Branching supports route stress around damage', x: 19, y: 18, tick: tick - 18, causal_parents: ['insight_41'] },
      { id: 'insight_52', author: 'agent_000007', title: 'A mineral gradient can nucleate repair at cracks', x: 34, y: 17, tick: tick - 7, causal_parents: ['insight_47'] },
    ],
    metrics: {
      artifact_score: 4.38,
      archive_entries: 13,
      deposited_insights: 18,
    },
  }
}

export function evolveDemo(previous: Snapshot): { snapshot: Snapshot; events: WorldEvent[] } {
  const tick = previous.tick + 1
  const agents = { ...previous.agents }
  agents.x = [...agents.x]
  agents.y = [...agents.y]
  agents.energy = agents.energy.map((value, index) => Math.max(0.2, Math.min(1, value + 0.008 * Math.sin(tick * 0.07 + index))))
  agents.inventory_total = agents.inventory_total.map((value, index) => Math.max(0, Math.min(12, value + 0.06 * Math.sin(tick * 0.11 + index * 0.8))))
  agents.last_action = agents.last_action.map((value, index) => tick % (9 + index % 7) === 0 ? [1, 2, 3, 5, 11, 13, 14][(tick + index) % 7] : value)

  const moved: number[] = []
  agents.x.forEach((value, index) => {
    if ((tick + index * 3) % 11 !== 0) return
    const phase = Math.floor((tick + index) / 11) % 4
    const dx = [1, 0, -1, 0][phase]
    const dy = [0, 1, 0, -1][phase]
    const nx = Math.max(2, Math.min(previous.world.width - 2, value + dx))
    const ny = Math.max(2, Math.min(previous.world.height - 2, agents.y[index] + dy))
    if (previous.world.terrain[ny * previous.world.width + nx] !== 0) {
      agents.x[index] = nx
      agents.y[index] = ny
      agents.last_action[index] = 1
      moved.push(index)
    }
  })

  const world = { ...previous.world }
  const day = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(tick / 26))
  world.solar = previous.world.solar.map((value, index) => {
    const shaded = previous.world.terrain[index] === 3 ? 0.55 : 1
    return day * shaded
  })
  world.temperature = previous.world.temperature.map((value, index) => value + 0.018 * (world.solar[index] - value))
  world.resource_mass = previous.world.resource_mass.map((value, index) => value > 0 ? Math.min(3.2, value + 0.002 * (3 - value)) : 0)

  const artifacts = { ...previous.artifacts }
  artifacts.maturity = artifacts.maturity.map((value) => Math.min(1, value + 0.0008))
  artifacts.open_fraction = artifacts.open_fraction.map((value, index) => Math.max(0.08, Math.min(1, value + 0.018 * Math.sin(tick * 0.09 + index))))
  artifacts.performance = artifacts.performance.map((value, index) => Math.max(0, Math.min(1, value + 0.002 * Math.sin(tick * 0.05 + index))))

  const events: WorldEvent[] = moved.length ? [{ tick, kind: 'agents_moved', payload: { indices: moved } }] : []
  if (tick % 13 === 0) events.push({ tick, kind: 'resource_harvested', payload: { agent: agents.ids[tick % agents.count], resource: 3 + (tick % 4) } })
  if (tick % 31 === 0) events.push({ tick, kind: 'message_delivered', payload: { sender: agents.ids[tick % agents.count], recipients: [agents.ids[(tick + 5) % agents.count], agents.ids[(tick + 13) % agents.count]] } })
  if (tick % 47 === 0) events.push({ tick, kind: 'insight_deposited', payload: { author: agents.ids[tick % agents.count], title: 'Observed coupled transport and structural response' } })

  return {
    snapshot: {
      ...previous,
      tick,
      world,
      agents,
      artifacts,
      metrics: {
        ...previous.metrics,
        artifact_score: artifacts.performance.reduce((sum, value) => sum + value, 0),
      },
    },
    events,
  }
}

export function initialDemoEvents(snapshot: Snapshot): WorldEvent[] {
  return [
    { tick: snapshot.tick - 7, kind: 'insight_deposited', payload: { author: 'agent_000007', title: 'Mineral gradients localize repair' } },
    { tick: snapshot.tick - 5, kind: 'message_delivered', payload: { sender: 'agent_000031', recipients: ['agent_000018', 'agent_000007'] } },
    { tick: snapshot.tick - 3, kind: 'artifact_program_installed', payload: { agent: 'agent_000018', artifact_id: 'artifact_00000002' } },
    { tick: snapshot.tick - 1, kind: 'resource_harvested', payload: { agent: 'agent_000011', resource: 3 } },
  ]
}
