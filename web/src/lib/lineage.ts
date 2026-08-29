import type { Snapshot, WorldEvent } from '../types'

export type LineageRelation =
  | 'causal'
  | 'authored'
  | 'observed'
  | 'tested'
  | 'built'
  | 'installed'
  | 'forked'
  | 'communicated'

export interface LineageNodeDetail {
  id: string
  label: string
  kind: 'insight' | 'evidence' | 'artifact' | 'agent' | 'program'
  author?: string
  tick?: number
  content?: string
  causalParents?: string[]
  artifactIndex?: number
  agentId?: string
  location?: [number, number]
  properties?: Record<string, string | number>
}

export interface LineageGraphLayout {
  positions: Record<string, { x: number; y: number }>
}

export interface LineageElement {
  data: {
    id: string
    label?: string
    kind?: 'insight' | 'evidence' | 'artifact' | 'agent' | 'program'
    source?: string
    target?: string
    communication?: boolean
    relation?: LineageRelation
    count?: number
    countLabel?: string
    detail?: LineageNodeDetail
  }
}

function nonEmptyString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const cleaned = value.trim()
  return cleaned || undefined
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map(nonEmptyString).filter((item): item is string => item !== undefined)
}

function recordLabel(record: Record<string, unknown>, fallback: string): string {
  const rawLabel = nonEmptyString(record.title)
    ?? nonEmptyString(record.claim)
    ?? nonEmptyString(record.content)
    ?? nonEmptyString(record.kind)
    ?? fallback
  const label = rawLabel.replace(/^agent_\d+\s*\|\s*/i, '')
  return label.length > 140 ? `${label.slice(0, 137)}…` : label
}

function recordAuthor(record: { author?: unknown; content?: unknown; claim?: unknown }) {
  const explicit = nonEmptyString(record.author)
  if (explicit) return explicit
  const content = nonEmptyString(record.content) ?? nonEmptyString(record.claim) ?? ''
  return content.match(/^(agent_\d+)\s*\|/i)?.[1]
}

function compactAgent(id: string) {
  return id.replace(/^agent_/, 'A').replace(/^A0+/, 'A')
}

function missingEvidenceLabel(id: string) {
  if (id.startsWith('observation_')) return 'Local observation'
  if (id.startsWith('test_')) return 'Material test'
  if (id.startsWith('recipe_')) return 'Material recipe'
  if (id.startsWith('fulfillment_')) return 'Request fulfillment'
  if (id.startsWith('message_')) return 'Agent message'
  return 'Causal evidence'
}

/**
 * Build a Cytoscape-safe graph from authoritative IDs and recorded provenance.
 * Every edge endpoint is guaranteed to exist and every element ID is non-empty.
 */
export function buildLineageElements(snapshot: Snapshot, events: WorldEvent[]): LineageElement[] {
  const nodes = new Map<string, LineageElement>()
  const edges = new Map<string, LineageElement>()
  const evidenceArtifactRelations: Array<[string, string]> = []
  let edgeSequence = 0

  const addNode = (
    id: string | undefined,
    label: string,
    kind: LineageElement['data']['kind'],
    detail?: Omit<LineageNodeDetail, 'id' | 'label' | 'kind'>,
  ) => {
    if (!id) return
    const existing = nodes.get(id)
    if (existing && existing.data.kind !== 'evidence') return
    nodes.set(id, {
      data: {
        id,
        label,
        kind,
        detail: kind ? { id, label, kind, ...detail } : undefined,
      },
    })
  }

  const addAgent = (id: string | undefined) => {
    if (!id) return
    addNode(id, compactAgent(id), 'agent', { agentId: id })
  }

  const addRelationEdge = (
    source: string | undefined,
    target: string | undefined,
    relation: LineageRelation,
    count = 1,
  ) => {
    if (!source || !target || source === target) return
    const pair = `${relation}\u0000${source}\u0000${target}`
    if (relation === 'causal' && !nodes.has(source)) {
      addNode(source, missingEvidenceLabel(source), 'evidence', {
        content: `A cited ${missingEvidenceLabel(source).toLowerCase()} whose compact metadata was not retained in this presentation snapshot.`,
      })
    }
    if (!nodes.has(target)) return
    const existing = edges.get(pair)
    const total = (existing?.data.count ?? 0) + count
    if (existing) {
      existing.data.count = total
      existing.data.countLabel = total > 1 ? `×${total}` : ''
      return
    }
    edges.set(pair, {
      data: {
        id: `${relation}-${edgeSequence++}`,
        source,
        target,
        relation,
        communication: relation === 'communicated',
        count: total,
        countLabel: total > 1 ? `×${total}` : '',
      },
    })
  }

  const addCausalEdge = (source: string | undefined, target: string | undefined) => {
    addRelationEdge(source, target, 'causal')
  }

  ;(snapshot.causal_evidence ?? []).forEach((record) => {
    const id = nonEmptyString(record.record_id)
    if (!id) return
    const author = recordAuthor(record)
    addNode(id, nonEmptyString(record.summary) ?? missingEvidenceLabel(id), 'evidence', {
      author,
      tick: record.tick,
      content: nonEmptyString(record.content) ?? nonEmptyString(record.summary),
      causalParents: stringList(record.causal_parents),
    })
    addAgent(author)
    addRelationEdge(
      author,
      id,
      record.kind === 'observation' ? 'observed' : record.kind === 'experiment_result' ? 'tested' : 'authored',
    )
    stringList(record.causal_parents).forEach((parent) => addCausalEdge(parent, id))
    stringList(record.related_artifacts).forEach((artifact) => {
      evidenceArtifactRelations.push([id, artifact])
    })
  })

  snapshot.archive.forEach((record, index) => {
    const id = nonEmptyString(record.record_id) ?? nonEmptyString(record.id) ?? `archive-${index}`
    const label = recordLabel(record, `Evidence ${index + 1}`)
    const author = nonEmptyString(record.author)
    addNode(id, label, 'evidence', {
      author,
      tick: typeof record.tick === 'number' ? record.tick : undefined,
      content: nonEmptyString(record.content) ?? nonEmptyString(record.title),
      causalParents: stringList(record.causal_parents),
    })
    addAgent(author)
    addRelationEdge(author, id, 'authored')
  })

  const insightIds = snapshot.insights.map((insight, index) => {
    const id = nonEmptyString(insight.record_id) ?? nonEmptyString(insight.id) ?? `insight-${index}`
    const label = recordLabel(insight, `Insight ${index + 1}`)
    const author = recordAuthor(insight)
    addNode(id, label, 'insight', {
      author,
      tick: typeof insight.tick === 'number' ? insight.tick : undefined,
      content: nonEmptyString(insight.content) ?? nonEmptyString(insight.claim) ?? nonEmptyString(insight.title),
      causalParents: stringList(insight.causal_parents),
      location: typeof insight.x === 'number' && typeof insight.y === 'number' ? [insight.x, insight.y] : undefined,
    })
    addAgent(author)
    addRelationEdge(author, id, 'authored')
    return id
  })

  snapshot.insights.forEach((insight, index) => {
    stringList(insight.causal_parents).forEach((parent) => addCausalEdge(parent, insightIds[index]))
  })

  ;(snapshot.program_catalog ?? []).forEach((program) => {
    addNode(program.program_id, program.name || program.program_id, 'program', {
      author: program.authors.join(', '),
      tick: program.first_tick,
      content: JSON.stringify(program.instructions, null, 2),
      properties: {
        'program ID': program.program_id,
        authors: program.authors.length,
        installations: program.installations.length,
        instructions: program.instructions.length,
      },
    })
    program.authors.forEach((author) => {
      addAgent(author)
      addRelationEdge(author, program.program_id, 'authored')
    })
  })

  ;(snapshot.program_lineage ?? []).forEach((edge) => {
    addRelationEdge(edge.parent_program_id, edge.child_program_id, 'forked')
    addAgent(edge.author)
    addRelationEdge(edge.author, edge.child_program_id, 'authored')
  })

  const artifactIds = snapshot.artifacts.ids.map((rawId, index) => {
    const id = nonEmptyString(rawId) ?? `artifact-${index}`
    const label = nonEmptyString(snapshot.artifacts.name[index])
      ?? nonEmptyString(snapshot.artifacts.program[index])
      ?? `Artifact ${index + 1}`
    const serviceSummary = Object.fromEntries(Object.entries(snapshot.artifacts.services).map(([name, values]) => [
      name.replaceAll('_', ' '),
      Number(values[index] ?? 0).toFixed(3),
    ]))
    const creator = nonEmptyString(snapshot.artifacts.creator?.[index])
    addNode(id, label, 'artifact', {
      author: creator,
      tick: snapshot.artifacts.created_tick?.[index],
      content: nonEmptyString(snapshot.artifacts.claimed_function[index])
        ?? nonEmptyString(snapshot.artifacts.architecture[index]),
      causalParents: stringList(snapshot.artifacts.causal_parents?.[index]),
      artifactIndex: index,
      location: [snapshot.artifacts.x[index], snapshot.artifacts.y[index]],
      properties: {
        performance: Number(snapshot.artifacts.performance[index] ?? 0).toFixed(3),
        'lifetime peak performance': Number(snapshot.artifacts.lifetime_peak_performance?.[index] ?? snapshot.artifacts.peak_performance[index] ?? 0).toFixed(3),
        health: `${Math.round(Number(snapshot.artifacts.health[index] ?? 0) * 100)}%`,
        program: snapshot.artifacts.program[index] || 'none',
        'program ID': snapshot.artifacts.program_id?.[index] || 'none',
        retired: snapshot.artifacts.retired?.[index] ? 'yes' : 'no',
        ...serviceSummary,
      },
    })
    addAgent(creator)
    addRelationEdge(creator, id, 'built')
    return id
  })

  artifactIds.forEach((artifactId, index) => {
    stringList(snapshot.artifacts.causal_parents?.[index]).forEach((parent) => {
      addCausalEdge(parent, artifactId)
    })
    addRelationEdge(snapshot.artifacts.program_id?.[index], artifactId, 'installed')
  })
  evidenceArtifactRelations.forEach(([evidence, artifact]) => {
    addRelationEdge(evidence, artifact, 'observed')
  })

  events.forEach((event) => {
    if (event.kind === 'artifact_built') {
      const artifactId = nonEmptyString(event.payload.artifact_id)
      const creator = nonEmptyString(event.payload.agent)
      const batch = event.payload.batch
      const parents = typeof batch === 'object' && batch !== null
        ? stringList((batch as Record<string, unknown>).causal_parents)
        : []
      if (artifactId && !nodes.has(artifactId)) {
        const label = nonEmptyString(event.payload.artifact_name) ?? artifactId
        addNode(artifactId, label, 'artifact', {
          author: creator,
          tick: event.tick,
          content: nonEmptyString(event.payload.claimed_function),
          causalParents: parents,
        })
      }
      parents.forEach((parent) => addCausalEdge(parent, artifactId))
      addAgent(creator)
      addRelationEdge(creator, artifactId, 'built')
      return
    }

    if (event.kind === 'artifact_program_installed') {
      const artifactId = nonEmptyString(event.payload.artifact_id)
      const programId = nonEmptyString(event.payload.program_id)
      const author = nonEmptyString(event.payload.agent)
      addAgent(author)
      addRelationEdge(author, programId, 'authored')
      addRelationEdge(programId, artifactId, 'installed')
      return
    }

    if (event.kind === 'sample_inspected' || event.kind === 'material_tested') {
      const recordId = nonEmptyString(event.payload.record_id)
      const author = nonEmptyString(event.payload.agent)
      if (!recordId || !author) return
      const isInspection = event.kind === 'sample_inspected'
      if (!nodes.has(recordId)) {
        const label = isInspection
          ? `Observation at ${event.payload.x ?? '?'}, ${event.payload.y ?? '?'}`
          : `Material test · utility ${Number(event.payload.material_utility ?? 0).toFixed(3)}`
        addNode(recordId, label, 'evidence', {
          author,
          tick: event.tick,
          content: JSON.stringify(event.payload, null, 2),
        })
      }
      addAgent(author)
      addRelationEdge(author, recordId, isInspection ? 'observed' : 'tested')
      if (isInspection) {
        const measurements = Array.isArray(event.payload.artifact_measurements)
          ? event.payload.artifact_measurements
          : []
        measurements.forEach((measurement) => {
          if (typeof measurement !== 'object' || measurement === null) return
          addRelationEdge(
            recordId,
            nonEmptyString((measurement as Record<string, unknown>).artifact_id),
            'observed',
          )
        })
      }
      return
    }

    if (event.kind !== 'message_delivered') return
    const sender = nonEmptyString(event.payload.sender)
    if (!sender) return
    stringList(event.payload.recipients).forEach((recipient) => {
      addAgent(sender)
      addAgent(recipient)
      addRelationEdge(sender, recipient, 'communicated')
    })
  })

  // This final guard makes malformed or partial server packets non-fatal to Cytoscape.
  const validEdges = [...edges.values()].filter((edge) => {
    const { source, target } = edge.data
    return Boolean(source && target && nodes.has(source) && nodes.has(target))
  })
  return [...nodes.values(), ...validEdges]
}
