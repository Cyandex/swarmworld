import { ACTION_NAMES, RESOURCE_NAMES, STATION_NAMES, TERRAIN_NAMES } from '../constants'
import type { FieldMode, Snapshot, WorldEvent } from '../types'
import {
  buildLineageElements,
  type LineageGraphLayout,
  type LineageNodeDetail,
  type LineageRelation,
} from './lineage'

export type PaperPanelTab = 'inspect' | 'activity' | 'lineage' | 'report'

export interface PaperArtifact {
  svg: string
  width: number
  height: number
  basename: string
}

export interface PanelPaperContext {
  tab: PaperPanelTab
  snapshot: Snapshot
  events: WorldEvent[]
  selectedAgentIndex: number
  selectedArtifactIndex: number
  selectedTile: [number, number] | null
  selectedLineageNode: LineageNodeDetail | null
}

const INK = '#172b26'
const MUTED = '#657a72'
const LINE = '#d6dfdb'
const GREEN = '#17846e'
const GOLD = '#b88720'
const BLUE = '#397aa8'
const CORAL = '#cb654e'

const TERRAIN_COLORS = [
  '#34596c', '#5f9da3', '#a9c79a', '#6f9578', '#b19daf',
  '#abc2c4', '#ad9875', '#938a92', '#c4b679',
]
const RESOURCE_COLORS = [
  '#ffffff', '#2a9d8f', '#d4a373', '#7f9f70', '#b58ec6',
  '#86ad69', '#8c7b68', '#5598c8', '#d9a441',
]

function escapeXml(value: unknown) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;')
}

function safeName(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

function short(value: unknown, maximum = 80) {
  const text = String(value ?? '').trim()
  return text.length > maximum ? `${text.slice(0, maximum - 1)}…` : text
}

function wrapLines(value: unknown, maximumCharacters: number, maximumLines = 10) {
  const words = String(value ?? '').replace(/\s+/g, ' ').trim().split(' ').filter(Boolean)
  const lines: string[] = []
  let line = ''
  words.forEach((word) => {
    const candidate = line ? `${line} ${word}` : word
    if (candidate.length <= maximumCharacters || !line) {
      line = candidate
    } else {
      lines.push(line)
      line = word
    }
  })
  if (line) lines.push(line)
  if (lines.length <= maximumLines) return lines
  const retained = lines.slice(0, maximumLines)
  retained[maximumLines - 1] = short(retained[maximumLines - 1], maximumCharacters - 1)
  return retained
}

function text(
  x: number,
  y: number,
  value: unknown,
  options: { size?: number; fill?: string; weight?: number; anchor?: string; family?: string } = {},
) {
  const { size = 18, fill = INK, weight = 400, anchor = 'start', family = 'Arial, Helvetica, sans-serif' } = options
  return `<text x="${x}" y="${y}" fill="${fill}" font-family="${family}" font-size="${size}" font-weight="${weight}" text-anchor="${anchor}">${escapeXml(value)}</text>`
}

function wrappedText(
  x: number,
  y: number,
  value: unknown,
  maximumCharacters: number,
  options: { size?: number; fill?: string; weight?: number; lineHeight?: number; maximumLines?: number } = {},
) {
  const size = options.size ?? 18
  const lineHeight = options.lineHeight ?? Math.round(size * 1.45)
  const lines = wrapLines(value, maximumCharacters, options.maximumLines ?? 10)
  const spans = lines.map((line, index) => (
    `<tspan x="${x}" dy="${index === 0 ? 0 : lineHeight}">${escapeXml(line)}</tspan>`
  )).join('')
  return {
    svg: `<text x="${x}" y="${y}" fill="${options.fill ?? INK}" font-family="Arial, Helvetica, sans-serif" font-size="${size}" font-weight="${options.weight ?? 400}">${spans}</text>`,
    nextY: y + Math.max(1, lines.length) * lineHeight,
  }
}

function documentSvg(
  width: number,
  height: number,
  titleValue: string,
  subtitle: string,
  body: string,
) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(titleValue)}">
  <rect width="${width}" height="${height}" fill="#ffffff"/>
  ${text(62, 58, titleValue, { size: 30, weight: 700 })}
  ${text(62, 88, subtitle, { size: 14, fill: MUTED })}
  <line x1="62" y1="106" x2="${width - 62}" y2="106" stroke="${GREEN}" stroke-width="3"/>
  ${body}
  ${text(width - 62, height - 28, 'SwarmWorld · authoritative state export', { size: 11, fill: MUTED, anchor: 'end' })}
</svg>`
}

function interpolateColor(start: string, end: string, amount: number) {
  const parse = (value: string) => [1, 3, 5].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16))
  const a = parse(start)
  const b = parse(end)
  const channel = (index: number) => Math.round(a[index] + (b[index] - a[index]) * Math.max(0, Math.min(1, amount)))
    .toString(16).padStart(2, '0')
  return `#${channel(0)}${channel(1)}${channel(2)}`
}

function fieldColor(field: FieldMode, value: number) {
  if (field === 'moisture') return interpolateColor('#f7fbff', '#2171b5', value)
  if (field === 'nutrients') return interpolateColor('#ffffe5', '#238443', value)
  if (field === 'contamination') return interpolateColor('#fff5f0', '#b30000', value)
  return interpolateColor('#fff7ec', '#d94801', value)
}

function scenarioCatalog(snapshot: Snapshot, kind: string) {
  return (snapshot.scenario ?? snapshot.world.scenario)?.catalogs?.[kind]
}

function catalogName(snapshot: Snapshot, kind: string, index: number, fallback: string[]) {
  const item = scenarioCatalog(snapshot, kind)?.[index]
  return String(item?.name ?? item?.id ?? fallback[index] ?? 'Unknown')
}

function catalogColor(snapshot: Snapshot, kind: string, index: number, fallback: string[]) {
  return String(scenarioCatalog(snapshot, kind)?.[index]?.color ?? fallback[index] ?? GOLD)
}

export function buildWorldPaperArtifact(
  snapshot: Snapshot,
  field: FieldMode,
  selectedAgent: string | null,
  selectedArtifact: string | null,
  selectedTile: [number, number] | null,
): PaperArtifact {
  const width = 1600
  const world = snapshot.world
  const cell = Math.min(1420 / world.width, 790 / world.height)
  const mapWidth = cell * world.width
  const mapHeight = cell * world.height
  const mapX = (width - mapWidth) / 2
  const mapY = 142
  const height = Math.ceil(mapY + mapHeight + 190)
  const parts: string[] = []
  const candidate = world[field as keyof typeof world]
  const values = field === 'terrain'
    ? world.terrain
    : world.fields?.[field] ?? (Array.isArray(candidate) ? candidate : [])

  parts.push(`<g shape-rendering="crispEdges">`)
  for (let y = 0; y < world.height; y += 1) {
    for (let x = 0; x < world.width; x += 1) {
      const index = y * world.width + x
      const fill = field === 'terrain'
        ? catalogColor(snapshot, 'terrains', Math.round(values[index] ?? 0), TERRAIN_COLORS)
        : fieldColor(field, Number(values[index] ?? 0))
      parts.push(`<rect x="${(mapX + x * cell).toFixed(2)}" y="${(mapY + y * cell).toFixed(2)}" width="${(cell + 0.15).toFixed(2)}" height="${(cell + 0.15).toFixed(2)}" fill="${fill}"/>`)
    }
  }
  parts.push('</g>')
  parts.push(`<rect x="${mapX}" y="${mapY}" width="${mapWidth}" height="${mapHeight}" fill="none" stroke="${INK}" stroke-width="1.5"/>`)

  if (field === 'terrain') {
    world.resource_kind.forEach((kind, index) => {
      if (!kind || Number(world.resource_mass[index] ?? 0) <= 0.02) return
      const x = index % world.width
      const y = Math.floor(index / world.width)
      parts.push(`<circle cx="${mapX + (x + 0.5) * cell}" cy="${mapY + (y + 0.5) * cell}" r="${Math.max(0.8, Math.min(2.8, cell * 0.16))}" fill="${catalogColor(snapshot, 'resources', kind, RESOURCE_COLORS)}" fill-opacity="0.72"/>`)
    })
  }

  world.stations.forEach((station, index) => {
    if (!station) return
    const x = index % world.width
    const y = Math.floor(index / world.width)
    const size = Math.max(5, cell * 0.58)
    parts.push(`<rect x="${mapX + (x + 0.5) * cell - size / 2}" y="${mapY + (y + 0.5) * cell - size / 2}" width="${size}" height="${size}" rx="1" fill="#ffffff" stroke="${INK}" stroke-width="1.4"/>`)
  })

  snapshot.agents.ids.forEach((id, index) => {
    const cx = mapX + (snapshot.agents.x[index] + 0.5) * cell
    const cy = mapY + (snapshot.agents.y[index] + 0.5) * cell
    const chosen = id === selectedAgent
    parts.push(`<circle cx="${cx}" cy="${cy}" r="${chosen ? 5.2 : 2.6}" fill="${chosen ? CORAL : '#263f38'}" stroke="#ffffff" stroke-width="${chosen ? 2 : 0.7}"/>`)
  })

  snapshot.artifacts.ids.forEach((id, index) => {
    const cx = mapX + (snapshot.artifacts.x[index] + 0.5) * cell
    const cy = mapY + (snapshot.artifacts.y[index] + 0.5) * cell
    const radius = id === selectedArtifact ? 8 : 5.5
    const label = snapshot.artifacts.name[index] || id
    parts.push(`<path d="M ${cx} ${cy - radius} L ${cx + radius} ${cy} L ${cx} ${cy + radius} L ${cx - radius} ${cy} Z" fill="${id === selectedArtifact ? CORAL : GREEN}" stroke="#ffffff" stroke-width="1.5"/>`)
    parts.push(text(cx + radius + 4, cy - 3, short(label, 38), { size: 10, fill: INK, weight: 600 }))
  })

  if (selectedTile) {
    parts.push(`<rect x="${mapX + selectedTile[0] * cell}" y="${mapY + selectedTile[1] * cell}" width="${cell}" height="${cell}" fill="none" stroke="${CORAL}" stroke-width="3"/>`)
  }

  const legendY = mapY + mapHeight + 43
  parts.push(text(62, legendY, `Field: ${field.replaceAll('_', ' ')}`, { size: 15, weight: 700, fill: GREEN }))
  parts.push(`<circle cx="245" cy="${legendY - 5}" r="4" fill="#263f38"/>${text(257, legendY, 'agent', { size: 13, fill: MUTED })}`)
  parts.push(`<path d="M 330 ${legendY - 11} L 336 ${legendY - 5} L 330 ${legendY + 1} L 324 ${legendY - 5} Z" fill="${GREEN}"/>${text(343, legendY, 'agent-designed artifact', { size: 13, fill: MUTED })}`)
  parts.push(`<rect x="520" y="${legendY - 11}" width="12" height="12" fill="#fff" stroke="${INK}"/>${text(540, legendY, 'distributed laboratory', { size: 13, fill: MUTED })}`)

  if (field === 'terrain') {
    const terrainItems = scenarioCatalog(snapshot, 'terrains') ?? TERRAIN_NAMES.map((name) => ({ name }))
    terrainItems.forEach((item, index) => {
      const name = String(item.name ?? ('id' in item ? item.id : `Terrain ${index}`))
      const x = 62 + (index % 5) * 215
      const y = legendY + 38 + Math.floor(index / 5) * 27
      parts.push(`<rect x="${x}" y="${y - 12}" width="15" height="15" fill="${catalogColor(snapshot, 'terrains', index, TERRAIN_COLORS)}" stroke="#aebbb5" stroke-width="0.5"/>${text(x + 23, y, name, { size: 12, fill: MUTED })}`)
    })
  } else {
    for (let step = 0; step <= 10; step += 1) {
      const x = 62 + step * 52
      parts.push(`<rect x="${x}" y="${legendY + 26}" width="53" height="16" fill="${fieldColor(field, step / 10)}"/>`)
    }
    parts.push(text(62, legendY + 61, '0.0', { size: 11, fill: MUTED }))
    parts.push(text(62 + 10 * 52 + 53, legendY + 61, '1.0', { size: 11, fill: MUTED, anchor: 'end' }))
  }

  const titleValue = `SwarmWorld society state · tick ${snapshot.tick}`
  const subtitle = `Seed ${snapshot.seed} · ${snapshot.agents.count} agents · ${snapshot.artifacts.count} artifacts · utility ${snapshot.metrics.artifact_score.toFixed(3)}`
  return {
    svg: documentSvg(width, height, titleValue, subtitle, parts.join('')),
    width,
    height,
    basename: `swarmworld-world-seed-${snapshot.seed}-tick-${snapshot.tick}`,
  }
}

function metricCards(items: Array<[string, string]>, width: number, startY: number) {
  const columns = Math.min(4, items.length)
  const gap = 14
  const cardWidth = (width - 124 - gap * (columns - 1)) / columns
  return items.map(([label, value], index) => {
    const column = index % columns
    const row = Math.floor(index / columns)
    const x = 62 + column * (cardWidth + gap)
    const y = startY + row * 94
    return `<rect x="${x}" y="${y}" width="${cardWidth}" height="76" rx="7" fill="#f5f8f6" stroke="${LINE}"/>
      ${text(x + 15, y + 25, label.toUpperCase(), { size: 11, fill: MUTED, weight: 700 })}
      ${text(x + 15, y + 55, value, { size: 22, fill: INK, weight: 700 })}`
  }).join('')
}

function sectionHeading(y: number, titleValue: string) {
  return `${text(62, y, titleValue.toUpperCase(), { size: 13, fill: GREEN, weight: 700 })}<line x1="62" y1="${y + 11}" x2="1138" y2="${y + 11}" stroke="${LINE}"/>`
}

function eventSummary(event: WorldEvent) {
  const payload = event.payload
  if (event.kind === 'agent_deliberated') return `${String(payload.agent ?? 'agent')} chose ${ACTION_NAMES[Number(payload.verb ?? 0)] ?? 'an action'}`
  if (event.kind === 'model_error') return `Model request failed for ${String(payload.agent ?? 'agent')}`
  if (event.kind === 'resource_harvested') return `${String(payload.agent ?? 'agent')} harvested ${RESOURCE_NAMES[Number(payload.resource ?? 0)] ?? 'material'}`
  if (event.kind === 'artifact_built') return `${String(payload.agent ?? 'agent')} built ${String(payload.artifact_name ?? 'an artifact')}`
  if (event.kind === 'message_delivered') return `${String(payload.sender ?? 'agent')} shared local knowledge`
  if (event.kind === 'insight_deposited') return `${String(payload.author ?? 'agent')} deposited an insight`
  return event.kind.replaceAll('_', ' ')
}

function buildInspectArtifact(context: PanelPaperContext): PaperArtifact {
  const { snapshot, selectedAgentIndex, selectedArtifactIndex, selectedTile } = context
  const width = 1200
  let height = 980
  const parts: string[] = []
  let titleValue = 'Society overview'
  let subtitle = `Seed ${snapshot.seed} · tick ${snapshot.tick}`

  if (selectedArtifactIndex >= 0) {
    const index = selectedArtifactIndex
    titleValue = snapshot.artifacts.name[index] || snapshot.artifacts.ids[index]
    subtitle = `Agent-designed material system · ${snapshot.artifacts.ids[index]} · tick ${snapshot.tick}`
    parts.push(metricCards([
      ['performance', Number(snapshot.artifacts.performance[index] ?? 0).toFixed(3)],
      ['lifetime peak', Number(snapshot.artifacts.lifetime_peak_performance?.[index] ?? snapshot.artifacts.peak_performance[index] ?? 0).toFixed(3)],
      ['health', `${Math.round(Number(snapshot.artifacts.health[index] ?? 0) * 100)}%`],
      ['maturity', `${Math.round(Number(snapshot.artifacts.maturity[index] ?? 0) * 100)}%`],
    ], width, 135))
    parts.push(sectionHeading(255, 'Scientific hypothesis'))
    let block = wrappedText(62, 292, snapshot.artifacts.claimed_function[index] || 'No claimed function recorded.', 102, { size: 18, maximumLines: 5 })
    parts.push(block.svg)
    block = wrappedText(62, block.nextY + 12, snapshot.artifacts.architecture[index] || 'No architecture description recorded.', 102, { size: 16, fill: MUTED, maximumLines: 5 })
    parts.push(block.svg)
    parts.push(text(62, block.nextY + 12, `${snapshot.scenario ?? snapshot.world.scenario ? 'Design inspiration' : 'Bio-inspiration'}: ${(snapshot.artifacts.bio_inspiration[index] ?? []).join(', ') || 'unspecified'}`, { size: 14, fill: GOLD, weight: 600 }))
    const servicesY = block.nextY + 62
    parts.push(sectionHeading(servicesY, 'Measured field services'))
    Object.entries(snapshot.artifacts.services).forEach(([name, values], serviceIndex) => {
      const y = servicesY + 42 + serviceIndex * 39
      const value = Math.max(0, Math.min(1, Number(values[index] ?? 0)))
      const peak = Math.max(0, Math.min(1, Number(snapshot.artifacts.peak_services[name]?.[index] ?? 0)))
      parts.push(text(62, y, name.replaceAll('_', ' '), { size: 14, fill: MUTED }))
      parts.push(`<rect x="270" y="${y - 15}" width="620" height="15" rx="3" fill="#edf2ef"/><rect x="270" y="${y - 15}" width="${620 * value}" height="15" rx="3" fill="${GREEN}"/><line x1="${270 + 620 * peak}" y1="${y - 20}" x2="${270 + 620 * peak}" y2="${y + 4}" stroke="${GOLD}" stroke-width="3"/>`)
      parts.push(text(915, y, `${value.toFixed(3)} · peak ${peak.toFixed(3)}`, { size: 13, fill: INK }))
    })
    const programY = servicesY + 42 + Object.keys(snapshot.artifacts.services).length * 39 + 32
    parts.push(sectionHeading(programY, 'Installed tick program'))
    const program = wrappedText(62, programY + 42, snapshot.artifacts.program[index] || 'No autonomous program installed.', 112, { size: 14, fill: INK, maximumLines: 12, lineHeight: 21 })
    parts.push(program.svg)
    height = Math.max(980, program.nextY + 90)
  } else if (selectedAgentIndex >= 0) {
    const index = selectedAgentIndex
    titleValue = snapshot.agents.ids[index].replace('agent_', 'Agent ')
    subtitle = `Agent state · seed ${snapshot.seed} · tick ${snapshot.tick}`
    parts.push(metricCards([
      ['position', `${snapshot.agents.x[index]}, ${snapshot.agents.y[index]}`],
      ['energy', `${Math.round(snapshot.agents.energy[index] * 100)}%`],
      ['inventory mass', Number(snapshot.agents.inventory_total[index]).toFixed(2)],
      ['current action', ACTION_NAMES[snapshot.agents.last_action[index]] ?? 'Unknown'],
    ], width, 135))
    parts.push(sectionHeading(255, 'Local authoritative observation'))
    const cellIndex = snapshot.agents.y[index] * snapshot.world.width + snapshot.agents.x[index]
    parts.push(metricCards([
      ['terrain', catalogName(snapshot, 'terrains', snapshot.world.terrain[cellIndex], TERRAIN_NAMES)],
      ['resource', catalogName(snapshot, 'resources', snapshot.world.resource_kind[cellIndex], RESOURCE_NAMES)],
      ['station', catalogName(snapshot, 'facilities', snapshot.world.stations[cellIndex], STATION_NAMES)],
      ['distance traveled', String(snapshot.agents.distance_traveled?.[index] ?? 0)],
      ['moisture', Number(snapshot.world.moisture[cellIndex]).toFixed(3)],
      ['nutrients', Number(snapshot.world.nutrients[cellIndex]).toFixed(3)],
      ['contamination', Number(snapshot.world.contamination[cellIndex]).toFixed(3)],
      ['temperature', Number(snapshot.world.temperature[cellIndex]).toFixed(3)],
    ], width, 285))
  } else if (selectedTile) {
    const cellIndex = selectedTile[1] * snapshot.world.width + selectedTile[0]
    titleValue = `World cell ${selectedTile[0]}, ${selectedTile[1]}`
    subtitle = `Environmental state · seed ${snapshot.seed} · tick ${snapshot.tick}`
    parts.push(metricCards([
      ['terrain', catalogName(snapshot, 'terrains', snapshot.world.terrain[cellIndex], TERRAIN_NAMES)],
      ['station', catalogName(snapshot, 'facilities', snapshot.world.stations[cellIndex], STATION_NAMES)],
      ['resource', catalogName(snapshot, 'resources', snapshot.world.resource_kind[cellIndex], RESOURCE_NAMES)],
      ['resource mass', Number(snapshot.world.resource_mass[cellIndex]).toFixed(3)],
      ['moisture', Number(snapshot.world.moisture[cellIndex]).toFixed(3)],
      ['nutrients', Number(snapshot.world.nutrients[cellIndex]).toFixed(3)],
      ['contamination', Number(snapshot.world.contamination[cellIndex]).toFixed(3)],
      ['temperature', Number(snapshot.world.temperature[cellIndex]).toFixed(3)],
    ], width, 135))
  } else {
    parts.push(metricCards([
      ['agents', String(snapshot.agents.count)],
      ['artifacts', String(snapshot.artifacts.count)],
      ['research score', `${Math.round((snapshot.research?.score ?? 0) * 100)}%`],
      ['artifact utility', snapshot.metrics.artifact_score.toFixed(3)],
    ], width, 135))
    parts.push(sectionHeading(255, 'Invented technological portfolio'))
    snapshot.artifacts.ids.slice(0, 18).forEach((id, index) => {
      const y = 295 + index * 31
      parts.push(text(62, y, short(snapshot.artifacts.name[index] || id, 58), { size: 15, weight: 600 }))
      parts.push(text(730, y, `performance ${Number(snapshot.artifacts.performance[index] ?? 0).toFixed(3)}`, { size: 13, fill: MUTED }))
      parts.push(text(980, y, `${snapshot.artifacts.x[index]}, ${snapshot.artifacts.y[index]}`, { size: 13, fill: MUTED }))
    })
    height = Math.max(820, 350 + Math.min(18, snapshot.artifacts.count) * 31)
  }

  return {
    svg: documentSvg(width, height, titleValue, subtitle, parts.join('')),
    width,
    height,
    basename: `swarmworld-inspect-seed-${snapshot.seed}-tick-${snapshot.tick}`,
  }
}

function buildActivityArtifact(context: PanelPaperContext): PaperArtifact {
  const width = 1200
  const retained = [...context.events].reverse().slice(0, 34)
  const height = Math.max(760, 190 + retained.length * 34)
  const parts: string[] = []
  retained.forEach((event, index) => {
    const y = 145 + index * 34
    parts.push(`<circle cx="76" cy="${y - 5}" r="5" fill="${event.kind.includes('artifact') || event.kind.includes('insight') ? GOLD : GREEN}"/>`)
    parts.push(text(97, y, `T${event.tick}`, { size: 13, fill: MUTED, weight: 700 }))
    parts.push(text(165, y, short(eventSummary(event), 108), { size: 14, fill: INK }))
    parts.push(text(1110, y, event.kind.replaceAll('_', ' '), { size: 10, fill: MUTED, anchor: 'end' }))
    parts.push(`<line x1="62" y1="${y + 13}" x2="1138" y2="${y + 13}" stroke="#edf1ef"/>`)
  })
  return {
    svg: documentSvg(width, height, 'Collective activity', `Seed ${context.snapshot.seed} · tick ${context.snapshot.tick} · ${context.events.length} retained events`, parts.join('')),
    width,
    height,
    basename: `swarmworld-activity-seed-${context.snapshot.seed}-tick-${context.snapshot.tick}`,
  }
}

function buildLineageArtifact(context: PanelPaperContext): PaperArtifact {
  const width = 1800
  const elements = buildLineageElements(context.snapshot, context.events.slice(-120))
  const nodes = elements.filter(({ data }) => !data.source && !data.target)
  const edges = elements.filter(({ data }) => data.source && data.target)
  const scientificEdges = edges.filter(({ data }) => data.relation !== 'communicated')
  const communicationEdges = edges.filter(({ data }) => data.relation === 'communicated')
  // A large society may draft many programs that never affect the world. Keep the
  // paper lineage focused on programs that were installed at least once and their
  // exact fork ancestors; the interactive graph remains the exhaustive catalog.
  const relevantProgramIds = new Set<string>(
    (context.snapshot.artifacts.program_id ?? []).filter((id) => Boolean(id)),
  )
  ;(context.snapshot.program_catalog ?? []).forEach((program) => {
    if (program.installations.length) relevantProgramIds.add(program.program_id)
  })
  let addedProgramAncestor = true
  while (addedProgramAncestor) {
    addedProgramAncestor = false
    scientificEdges.forEach(({ data }) => {
      if (data.relation !== 'forked' || !data.source || !data.target) return
      if (relevantProgramIds.has(data.target) && !relevantProgramIds.has(data.source)) {
        relevantProgramIds.add(data.source)
        addedProgramAncestor = true
      }
    })
  }
  const programNodeIds = new Set(
    nodes.filter(({ data }) => data.kind === 'program').map(({ data }) => data.id),
  )
  const paperScientificEdges = scientificEdges.filter(({ data }) => (
    (!data.source || !programNodeIds.has(data.source) || relevantProgramIds.has(data.source))
    && (!data.target || !programNodeIds.has(data.target) || relevantProgramIds.has(data.target))
  ))
  const coreIds = new Set<string>()
  paperScientificEdges.forEach(({ data }) => {
    if (data.source) coreIds.add(data.source)
    if (data.target) coreIds.add(data.target)
  })
  nodes.forEach(({ data }) => {
    if (data.kind === 'artifact' || (data.kind === 'program' && relevantProgramIds.has(data.id))) {
      coreIds.add(data.id)
    }
  })
  const coreNodes = nodes.filter(({ data }) => coreIds.has(data.id))
  const byId = new Map(nodes.map((node) => [node.data.id, node]))
  const laneOrder = ['agent', 'evidence', 'insight', 'program', 'artifact'] as const
  const laneConfig = {
    agent: { title: 'Scientific actors', x: 155, width: 190, color: BLUE, fill: '#f2f7fa' },
    evidence: { title: 'Causal observations & tests', x: 455, width: 270, color: '#9e8351', fill: '#faf7ef' },
    insight: { title: 'Published insights', x: 790, width: 285, color: GOLD, fill: '#fff9e9' },
    program: { title: 'Executable programs', x: 1150, width: 285, color: '#9368b3', fill: '#f8f2fb' },
    artifact: { title: 'Invented artifacts', x: 1550, width: 350, color: GREEN, fill: '#eef8f4' },
  }
  const lanes = new Map<string, typeof coreNodes>()
  laneOrder.forEach((kind) => lanes.set(kind, coreNodes.filter(({ data }) => data.kind === kind)))
  const nodeTick = (id: string) => {
    const node = byId.get(id)
    if (typeof node?.data.detail?.tick === 'number') return node.data.detail.tick
    const targetTicks = paperScientificEdges
      .filter(({ data }) => data.source === id)
      .map(({ data }) => data.target ? byId.get(data.target)?.data.detail?.tick : undefined)
      .filter((tick): tick is number => typeof tick === 'number')
    return targetTicks.length ? Math.min(...targetTicks) : Number.MAX_SAFE_INTEGER
  }
  lanes.forEach((lane) => lane.sort((a, b) => (
    nodeTick(a.data.id) - nodeTick(b.data.id)
    || (a.data.label ?? a.data.id).localeCompare(b.data.label ?? b.data.id)
  )))
  const largestLane = Math.max(1, ...[...lanes.values()].map((lane) => lane.length))
  const laneTop = 190
  const laneHeight = Math.max(760, largestLane * 52)
  const laneBottom = laneTop + laneHeight
  const communicationPairs = [...communicationEdges]
    .sort((a, b) => (b.data.count ?? 1) - (a.data.count ?? 1))
    .slice(0, 12)
  const communicationTop = laneBottom + 72
  const communicationHeight = communicationPairs.length ? 220 : 110
  const height = communicationTop + communicationHeight + 125
  const positions = new Map<string, { x: number; y: number; width: number; height: number }>()

  lanes.forEach((lane, kind) => {
    const config = laneConfig[kind as keyof typeof laneConfig]
    lane.forEach((node, index) => {
      const y = laneTop + 58 + (laneHeight - 96) * (index + 0.5) / Math.max(1, lane.length)
      positions.set(node.data.id, { x: config.x, y, width: config.width, height: kind === 'artifact' ? 42 : 34 })
    })
  })

  const relationStyle: Record<string, { color: string; width: number; dash?: string }> = {
    causal: { color: '#9aaba4', width: 1.35 },
    authored: { color: '#6b99b8', width: 1.2, dash: '3 5' },
    observed: { color: '#3f7fa9', width: 1.5 },
    tested: { color: '#b88720', width: 1.7 },
    built: { color: GREEN, width: 2.5 },
    installed: { color: '#9368b3', width: 2.1 },
    forked: { color: '#9368b3', width: 1.8, dash: '7 5' },
  }
  const markerDefs = Object.entries(relationStyle).map(([relation, style]) => (
    `<marker id="arrow-${relation}" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="${style.color}"/></marker>`
  )).join('')
  const parts: string[] = [`<defs>${markerDefs}</defs>`]

  laneOrder.forEach((kind) => {
    const config = laneConfig[kind]
    const count = lanes.get(kind)?.length ?? 0
    parts.push(`<rect x="${config.x - config.width / 2 - 18}" y="${laneTop - 28}" width="${config.width + 36}" height="${laneHeight + 50}" rx="14" fill="${config.fill}" stroke="${LINE}"/>`)
    parts.push(text(config.x, laneTop, config.title.toUpperCase(), { size: 12, fill: config.color, weight: 700, anchor: 'middle' }))
    parts.push(text(config.x, laneTop + 20, `${count} node${count === 1 ? '' : 's'}`, { size: 10, fill: MUTED, anchor: 'middle' }))
  })

  paperScientificEdges.forEach(({ data }) => {
    if (!data.source || !data.target) return
    const source = positions.get(data.source)
    const target = positions.get(data.target)
    if (!source || !target) return
    const relation = data.relation ?? 'causal'
    const style = relationStyle[relation] ?? relationStyle.causal
    let path = ''
    if (Math.abs(target.x - source.x) < 10) {
      const bend = source.x + source.width / 2 + 34 + Math.min(70, Math.abs(target.y - source.y) * 0.12)
      path = `M ${source.x + source.width / 2} ${source.y} C ${bend} ${source.y}, ${bend} ${target.y}, ${target.x + target.width / 2} ${target.y}`
    } else {
      const forward = target.x > source.x
      const x1 = source.x + (forward ? source.width / 2 : -source.width / 2)
      const x2 = target.x + (forward ? -target.width / 2 : target.width / 2)
      const control = Math.max(42, Math.abs(x2 - x1) * 0.42)
      path = `M ${x1} ${source.y} C ${x1 + (forward ? control : -control)} ${source.y}, ${x2 + (forward ? -control : control)} ${target.y}, ${x2} ${target.y}`
    }
    parts.push(`<path d="${path}" fill="none" stroke="${style.color}" stroke-width="${style.width}" ${style.dash ? `stroke-dasharray="${style.dash}"` : ''} marker-end="url(#arrow-${relation})" opacity="0.58"/>`)
  })

  laneOrder.forEach((kind) => {
    const config = laneConfig[kind]
    ;(lanes.get(kind) ?? []).forEach(({ data }) => {
      const position = positions.get(data.id)
      if (!position) return
      const left = position.x - position.width / 2
      const top = position.y - position.height / 2
      parts.push(`<rect x="${left}" y="${top}" width="${position.width}" height="${position.height}" rx="${kind === 'agent' ? 17 : 6}" fill="#ffffff" stroke="${config.color}" stroke-width="${kind === 'artifact' ? 2 : 1.2}"/>`)
      parts.push(`<rect x="${left}" y="${top}" width="6" height="${position.height}" rx="3" fill="${config.color}"/>`)
      const label = kind === 'agent' ? (data.label ?? data.id) : short(data.label ?? data.id, kind === 'artifact' ? 42 : 32)
      parts.push(text(left + 16, position.y + 4, label, { size: kind === 'artifact' ? 12 : 10.5, fill: INK, weight: kind === 'artifact' ? 700 : 500 }))
      if (typeof data.detail?.tick === 'number') {
        parts.push(text(left + position.width - 10, position.y + 4, `T${data.detail.tick}`, { size: 9, fill: MUTED, anchor: 'end' }))
      }
    })
  })

  parts.push(text(62, communicationTop, 'RECENT AGENT EXCHANGES · SEPARATE FROM CAUSAL PROVENANCE', { size: 13, fill: GREEN, weight: 700 }))
  parts.push(`<line x1="62" y1="${communicationTop + 11}" x2="${width - 62}" y2="${communicationTop + 11}" stroke="${LINE}"/>`)
  if (communicationPairs.length) {
    communicationPairs.forEach(({ data }, index) => {
      const column = index % 4
      const row = Math.floor(index / 4)
      const boxWidth = 398
      const x = 62 + column * 421
      const y = communicationTop + 34 + row * 52
      const sender = byId.get(data.source ?? '')?.data.label ?? data.source ?? 'agent'
      const recipient = byId.get(data.target ?? '')?.data.label ?? data.target ?? 'agent'
      parts.push(`<rect x="${x}" y="${y}" width="${boxWidth}" height="38" rx="7" fill="#f1faf7" stroke="#b9dcd1"/>`)
      parts.push(text(x + 14, y + 24, `${sender}  →  ${recipient}`, { size: 12, fill: INK, weight: 600 }))
      parts.push(text(x + boxWidth - 14, y + 24, `×${data.count ?? 1}`, { size: 12, fill: GREEN, weight: 700, anchor: 'end' }))
    })
    parts.push(text(62, communicationTop + 204, 'Counts use the retained event window; communication edges do not imply causal contribution.', { size: 11, fill: MUTED }))
  } else {
    parts.push(text(62, communicationTop + 48, 'No message-delivery events are present in the retained event window for this displayed state.', { size: 13, fill: MUTED }))
  }

  const legendY = height - 72
  ;([['causal', '#9aaba4'], ['authored', '#6b99b8'], ['observed', '#3f7fa9'], ['tested', '#b88720'], ['built', GREEN], ['installed / forked', '#9368b3']] as Array<[string, string]>).forEach(([name, color], index) => {
    const x = 62 + index * 275
    parts.push(`<line x1="${x}" y1="${legendY - 4}" x2="${x + 30}" y2="${legendY - 4}" stroke="${color}" stroke-width="3"/>${text(x + 40, legendY, name, { size: 11, fill: MUTED })}`)
  })
  return {
    svg: documentSvg(width, height, 'Scientific interaction & technology lineage', `Seed ${context.snapshot.seed} · tick ${context.snapshot.tick} · ${coreNodes.length} scientific nodes · ${paperScientificEdges.length} typed relations`, parts.join('')),
    width,
    height,
    basename: `swarmworld-lineage-seed-${context.snapshot.seed}-tick-${context.snapshot.tick}`,
  }
}

const NETWORK_RELATION_STYLE: Record<LineageRelation, { color: string; width: number; dash?: string }> = {
  causal: { color: '#91a39c', width: 1.15 },
  authored: { color: '#6b99b8', width: 1.1, dash: '3 5' },
  observed: { color: '#397aa8', width: 1.4 },
  tested: { color: '#b88720', width: 1.6 },
  built: { color: '#17846e', width: 2.3 },
  installed: { color: '#9368b3', width: 1.9 },
  forked: { color: '#9368b3', width: 1.65, dash: '7 5' },
  communicated: { color: '#3b9c83', width: 1.3, dash: '6 6' },
}

function stableCurveOffset(value: string) {
  let hash = 0
  for (let index = 0; index < value.length; index += 1) {
    hash = ((hash << 5) - hash + value.charCodeAt(index)) | 0
  }
  return ((Math.abs(hash) % 9) - 4) * 3.5
}

/**
 * Rebuild the interactive force-directed lineage as editable white-background SVG.
 * Node coordinates come from Cytoscape, while all marks and labels are regenerated
 * from authoritative graph data rather than raster-capturing the dark UI canvas.
 */
export function buildLineageNetworkPaperArtifact(
  context: PanelPaperContext,
  layout: LineageGraphLayout,
): PaperArtifact {
  const width = 1800
  const elements = buildLineageElements(context.snapshot, context.events.slice(-120))
  const nodes = elements.filter(({ data }) => (
    !data.source && !data.target && layout.positions[data.id]
  ))
  const nodeIds = new Set(nodes.map(({ data }) => data.id))
  const edges = elements.filter(({ data }) => (
    data.source && data.target && nodeIds.has(data.source) && nodeIds.has(data.target)
  ))
  if (!nodes.length) throw new Error('The force-directed lineage layout is not ready yet')

  const rawPositions = nodes.map(({ data }) => layout.positions[data.id])
  const minimumX = Math.min(...rawPositions.map(({ x }) => x))
  const maximumX = Math.max(...rawPositions.map(({ x }) => x))
  const minimumY = Math.min(...rawPositions.map(({ y }) => y))
  const maximumY = Math.max(...rawPositions.map(({ y }) => y))
  const sourceWidth = Math.max(1, maximumX - minimumX)
  const sourceHeight = Math.max(1, maximumY - minimumY)
  const graphX = 88
  const graphY = 150
  const graphWidth = width - 2 * graphX
  const graphHeight = Math.round(Math.max(920, Math.min(2200, graphWidth * sourceHeight / sourceWidth)))
  const horizontalPadding = 72
  const verticalPadding = 72
  const scale = Math.min(
    (graphWidth - 2 * horizontalPadding) / sourceWidth,
    (graphHeight - 2 * verticalPadding) / sourceHeight,
  )
  const usedWidth = sourceWidth * scale
  const usedHeight = sourceHeight * scale
  const offsetX = graphX + (graphWidth - usedWidth) / 2
  const offsetY = graphY + (graphHeight - usedHeight) / 2
  const position = (id: string) => {
    const raw = layout.positions[id]
    return {
      x: offsetX + (raw.x - minimumX) * scale,
      y: offsetY + (raw.y - minimumY) * scale,
    }
  }
  const legendTop = graphY + graphHeight + 48
  const height = legendTop + 190
  const fontSize = nodes.length > 180 ? 7 : nodes.length > 100 ? 8 : 9.5
  const degree = new Map<string, number>()
  edges.forEach(({ data }) => {
    if (data.source) degree.set(data.source, (degree.get(data.source) ?? 0) + 1)
    if (data.target) degree.set(data.target, (degree.get(data.target) ?? 0) + 1)
  })
  const markerDefs = Object.entries(NETWORK_RELATION_STYLE).map(([relation, style]) => (
    `<marker id="network-arrow-${relation}" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="${style.color}"/></marker>`
  )).join('')
  const parts: string[] = [
    `<defs>${markerDefs}</defs>`,
    `<rect x="62" y="132" width="${width - 124}" height="${graphHeight + 36}" rx="14" fill="#fbfdfc" stroke="${LINE}"/>`,
  ]

  edges.forEach(({ data }) => {
    if (!data.source || !data.target) return
    const source = position(data.source)
    const target = position(data.target)
    const dx = target.x - source.x
    const dy = target.y - source.y
    const distance = Math.max(1, Math.hypot(dx, dy))
    const curve = stableCurveOffset(`${data.relation}:${data.source}:${data.target}`)
    const controlX = (source.x + target.x) / 2 - dy / distance * curve
    const controlY = (source.y + target.y) / 2 + dx / distance * curve
    const relation = data.relation ?? 'causal'
    const style = NETWORK_RELATION_STYLE[relation]
    parts.push(`<path d="M ${source.x.toFixed(2)} ${source.y.toFixed(2)} Q ${controlX.toFixed(2)} ${controlY.toFixed(2)} ${target.x.toFixed(2)} ${target.y.toFixed(2)}" fill="none" stroke="${style.color}" stroke-width="${style.width}" ${style.dash ? `stroke-dasharray="${style.dash}"` : ''} marker-end="url(#network-arrow-${relation})" opacity="${relation === 'communicated' ? 0.48 : 0.56}"/>`)
    if ((data.count ?? 1) > 1) {
      parts.push(text(controlX, controlY - 4, `×${data.count}`, { size: 8, fill: style.color, weight: 700, anchor: 'middle' }))
    }
  })

  const nodeColor = {
    insight: GOLD,
    evidence: '#9e8351',
    artifact: GREEN,
    agent: BLUE,
    program: '#9368b3',
  }
  nodes
    .slice()
    .sort((a, b) => (degree.get(a.data.id) ?? 0) - (degree.get(b.data.id) ?? 0))
    .forEach(({ data }) => {
      const point = position(data.id)
      const kind = data.kind ?? 'evidence'
      const color = nodeColor[kind]
      if (context.selectedLineageNode?.id === data.id) {
        parts.push(`<circle cx="${point.x}" cy="${point.y}" r="18" fill="none" stroke="#d8664b" stroke-width="2.5"/>`)
      }
      if (kind === 'artifact') {
        parts.push(`<path d="M ${point.x} ${point.y - 12} L ${point.x + 12} ${point.y} L ${point.x} ${point.y + 12} L ${point.x - 12} ${point.y} Z" fill="${color}" stroke="#ffffff" stroke-width="2"/>`)
      } else if (kind === 'program') {
        parts.push(`<path d="M ${point.x - 10} ${point.y - 8} L ${point.x + 10} ${point.y - 8} L ${point.x + 14} ${point.y} L ${point.x + 10} ${point.y + 8} L ${point.x - 10} ${point.y + 8} L ${point.x - 14} ${point.y} Z" fill="${color}" stroke="#ffffff" stroke-width="2"/>`)
      } else if (kind === 'evidence') {
        parts.push(`<rect x="${point.x - 8}" y="${point.y - 8}" width="16" height="16" rx="2" fill="${color}" stroke="#ffffff" stroke-width="1.6"/>`)
      } else {
        parts.push(`<circle cx="${point.x}" cy="${point.y}" r="${kind === 'agent' ? 7 : 9}" fill="${color}" stroke="#ffffff" stroke-width="1.8"/>`)
      }
      const maximumLabel = nodes.length > 160 ? 24 : 34
      const label = short(data.label ?? data.id, maximumLabel)
      parts.push(`<text x="${point.x}" y="${point.y + 22}" fill="${INK}" font-family="Arial, Helvetica, sans-serif" font-size="${fontSize}" font-weight="${kind === 'artifact' ? 700 : 500}" text-anchor="middle" style="paint-order:stroke;stroke:#ffffff;stroke-width:3px;stroke-linejoin:round">${escapeXml(label)}</text>`)
    })

  const nodeLegend = [
    ['insight', GOLD, 'circle'], ['evidence', '#9e8351', 'square'],
    ['artifact', GREEN, 'diamond'], ['agent', BLUE, 'circle'], ['program', '#9368b3', 'hexagon'],
  ] as const
  nodeLegend.forEach(([name, color, shape], index) => {
    const x = 72 + index * 205
    const y = legendTop + 20
    if (shape === 'square') parts.push(`<rect x="${x}" y="${y - 8}" width="14" height="14" rx="2" fill="${color}"/>`)
    else if (shape === 'diamond') parts.push(`<path d="M ${x + 7} ${y - 9} L ${x + 16} ${y} L ${x + 7} ${y + 9} L ${x - 2} ${y} Z" fill="${color}"/>`)
    else if (shape === 'hexagon') parts.push(`<path d="M ${x} ${y - 7} L ${x + 14} ${y - 7} L ${x + 18} ${y} L ${x + 14} ${y + 7} L ${x} ${y + 7} L ${x - 4} ${y} Z" fill="${color}"/>`)
    else parts.push(`<circle cx="${x + 7}" cy="${y}" r="7" fill="${color}"/>`)
    parts.push(text(x + 26, y + 4, name, { size: 11, fill: MUTED }))
  })
  ;(['causal', 'authored', 'observed', 'tested', 'built', 'installed', 'forked', 'communicated'] as LineageRelation[]).forEach((relation, index) => {
    const style = NETWORK_RELATION_STYLE[relation]
    const x = 72 + index * 205
    const y = legendTop + 68
    parts.push(`<line x1="${x}" y1="${y}" x2="${x + 30}" y2="${y}" stroke="${style.color}" stroke-width="${style.width + 0.8}" ${style.dash ? `stroke-dasharray="${style.dash}"` : ''}/>`)
    parts.push(text(x + 40, y + 4, relation, { size: 10, fill: MUTED }))
  })
  parts.push(text(72, legendTop + 118, 'Node positions preserve the interactive force-directed layout; layout geometry is presentation-only.', { size: 11, fill: MUTED }))

  return {
    svg: documentSvg(
      width,
      height,
      'Force-directed scientific society graph',
      `Seed ${context.snapshot.seed} · tick ${context.snapshot.tick} · ${nodes.length} nodes · ${edges.length} typed relations`,
      parts.join(''),
    ),
    width,
    height,
    basename: `swarmworld-lineage-graph-seed-${context.snapshot.seed}-tick-${context.snapshot.tick}`,
  }
}

function buildReportArtifact(context: PanelPaperContext): PaperArtifact {
  const { snapshot } = context
  const width = 1200
  const inventions = snapshot.research?.inventions ?? []
  const height = Math.max(1040, 720 + inventions.slice(0, 12).length * 34)
  const parts: string[] = []
  const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length)
  parts.push(metricCards([
    ['agents', String(snapshot.agents.count)],
    ['artifacts', String(snapshot.artifacts.count)],
    ['research score', `${Math.round((snapshot.research?.score ?? 0) * 100)}%`],
    ['utility', snapshot.metrics.artifact_score.toFixed(3)],
    ['mean energy', `${Math.round(mean(snapshot.agents.energy) * 100)}%`],
    ['mean moisture', mean(snapshot.world.moisture).toFixed(3)],
    ['contamination', mean(snapshot.world.contamination).toFixed(3)],
    ['archive records', String(snapshot.metrics.archive_entries)],
  ], width, 135))
  parts.push(sectionHeading(345, 'Collective science milestones'))
  Object.entries(snapshot.research?.milestones ?? {}).forEach(([name, reached], index) => {
    const column = index % 2
    const row = Math.floor(index / 2)
    const x = 62 + column * 538
    const y = 385 + row * 27
    parts.push(`<circle cx="${x + 5}" cy="${y - 5}" r="5" fill="${reached ? GREEN : '#dce4e0'}" stroke="${reached ? GREEN : '#9aaca4'}"/>`)
    parts.push(text(x + 20, y, name.replaceAll('_', ' '), { size: 13, fill: reached ? INK : MUTED }))
  })
  const milestoneRows = Math.ceil(Object.keys(snapshot.research?.milestones ?? {}).length / 2)
  const inventionY = 415 + milestoneRows * 27
  parts.push(sectionHeading(inventionY, 'Validated inventions'))
  inventions.slice(0, 12).forEach((invention, index) => {
    const y = inventionY + 42 + index * 34
    parts.push(text(62, y, short(invention.name, 52), { size: 15, fill: INK, weight: 600 }))
    parts.push(text(720, y, short(invention.claimed_function, 46), { size: 12, fill: MUTED }))
    parts.push(text(1110, y, invention.performance.toFixed(3), { size: 13, fill: GREEN, weight: 700, anchor: 'end' }))
    parts.push(`<line x1="62" y1="${y + 12}" x2="1138" y2="${y + 12}" stroke="#edf1ef"/>`)
  })
  if (!inventions.length) parts.push(text(62, inventionY + 48, 'No validated inventions recorded at this tick.', { size: 15, fill: MUTED }))
  return {
    svg: documentSvg(width, height, 'Deterministic situation report', `Seed ${snapshot.seed} · tick ${snapshot.tick} · generated without an LLM`, parts.join('')),
    width,
    height,
    basename: `swarmworld-report-seed-${snapshot.seed}-tick-${snapshot.tick}`,
  }
}

export function buildPanelPaperArtifact(context: PanelPaperContext): PaperArtifact {
  if (context.tab === 'activity') return buildActivityArtifact(context)
  if (context.tab === 'lineage') return buildLineageArtifact(context)
  if (context.tab === 'report') return buildReportArtifact(context)
  return buildInspectArtifact(context)
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

async function rasterizeSvg(svg: string, width: number, height: number) {
  const source = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' })
  const sourceUrl = URL.createObjectURL(source)
  try {
    const image = new Image()
    image.decoding = 'async'
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve()
      image.onerror = () => reject(new Error('Unable to rasterize the SVG snapshot'))
      image.src = sourceUrl
    })
    const scale = 2
    const canvas = document.createElement('canvas')
    canvas.width = width * scale
    canvas.height = height * scale
    const context = canvas.getContext('2d')
    if (!context) throw new Error('Canvas rendering is unavailable')
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, canvas.width, canvas.height)
    context.scale(scale, scale)
    context.drawImage(image, 0, 0, width, height)
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob(
      (blob) => blob ? resolve(blob) : reject(new Error('Unable to encode PNG snapshot')),
      'image/png',
    ))
  } finally {
    URL.revokeObjectURL(sourceUrl)
  }
}

export async function downloadPaperArtifact(artifact: PaperArtifact) {
  const basename = safeName(artifact.basename) || 'swarmworld-snapshot'
  const png = await rasterizeSvg(artifact.svg, artifact.width, artifact.height)
  downloadBlob(new Blob([artifact.svg], { type: 'image/svg+xml;charset=utf-8' }), `${basename}.svg`)
  downloadBlob(png, `${basename}.png`)
}
