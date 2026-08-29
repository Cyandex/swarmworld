import cytoscape from 'cytoscape'
import { describe, expect, it } from 'vitest'
import { createDemoSnapshot } from './demo'
import { buildLineageElements } from './lineage'

describe('lineage graph construction', () => {
  it('uses server record_id values and materializes missing parent nodes', () => {
    const snapshot = createDemoSnapshot(17, 20)
    snapshot.archive = []
    snapshot.insights = [{
      record_id: 'insight-child',
      content: 'A measured result',
      causal_parents: ['private-observation'],
    }]

    const elements = buildLineageElements(snapshot, [])
    const nodes = elements.filter(({ data }) => !data.source)
    const edges = elements.filter(({ data }) => data.source)

    expect(nodes.some(({ data }) => data.id === 'insight-child')).toBe(true)
    expect(nodes.some(({ data }) => data.id === 'private-observation')).toBe(true)
    expect(edges.some(({ data }) => data.source === 'private-observation' && data.target === 'insight-child')).toBe(true)
    expect(nodes.find(({ data }) => data.id === 'insight-child')?.data.detail).toMatchObject({
      id: 'insight-child',
      kind: 'insight',
      content: 'A measured result',
      causalParents: ['private-observation'],
    })

    const graph = cytoscape({ headless: true, elements })
    expect(graph.nodes()).toHaveLength(nodes.length)
    graph.destroy()
  })

  it('rejects malformed communication endpoints instead of emitting invalid edges', () => {
    const snapshot = createDemoSnapshot(17, 20)
    snapshot.archive = []
    snapshot.insights = []
    snapshot.artifacts.ids = []

    const elements = buildLineageElements(snapshot, [{
      tick: 20,
      kind: 'message_delivered',
      payload: { sender: '', recipients: ['agent_000001'] },
    }])

    expect(elements.some(({ data }) => data.source !== undefined)).toBe(false)
  })

  it('links artifacts only through recorded causal parents', () => {
    const snapshot = createDemoSnapshot(17, 20)
    snapshot.archive = []
    snapshot.insights = [{ record_id: 'insight-a', content: 'Evidence' }]
    snapshot.artifacts.ids = ['artifact-a']
    snapshot.artifacts.name = ['Invented material']
    snapshot.artifacts.program = ['']
    snapshot.artifacts.causal_parents = [['insight-a']]

    const elements = buildLineageElements(snapshot, [])
    expect(elements.some(({ data }) => data.source === 'insight-a' && data.target === 'artifact-a')).toBe(true)
    expect(elements.find(({ data }) => data.id === 'artifact-a')?.data.detail).toMatchObject({
      id: 'artifact-a',
      kind: 'artifact',
      label: 'Invented material',
      causalParents: ['insight-a'],
    })
  })

  it('renders clickable program descent and installation edges', () => {
    const snapshot = createDemoSnapshot(17, 20)
    snapshot.artifacts.ids = ['artifact_00000000']
    snapshot.artifacts.program = ['child']
    snapshot.artifacts.program_id = ['program_child']
    snapshot.program_catalog = [
      { program_id: 'program_parent', name: 'parent', instructions: [{ op: 'const' }], authors: ['agent_000000'], first_tick: 1, installations: [] },
      { program_id: 'program_child', name: 'child', instructions: [{ op: 'collect_water' }], authors: ['agent_000001'], first_tick: 2, installations: [] },
    ]
    snapshot.program_lineage = [{ parent_program_id: 'program_parent', child_program_id: 'program_child', tick: 2, artifact_id: 'artifact_00000000', author: 'agent_000001', instruction_diff: [] }]

    const elements = buildLineageElements(snapshot, [])
    expect(elements.some(({ data }) => data.id === 'program_parent' && data.kind === 'program')).toBe(true)
    expect(elements.some(({ data }) => data.source === 'program_parent' && data.target === 'program_child' && data.relation === 'forked')).toBe(true)
    expect(elements.some(({ data }) => data.source === 'program_child' && data.target === 'artifact_00000000' && data.relation === 'installed')).toBe(true)
  })

  it('makes authorship, evidence production, construction, and repeated communication explicit', () => {
    const snapshot = createDemoSnapshot(17, 20)
    snapshot.archive = []
    snapshot.insights = [{
      record_id: 'insight-a',
      author: 'agent_000001',
      content: 'agent_000001 | A concise public finding.',
      causal_parents: ['observation-a'],
      tick: 7,
    }]
    snapshot.causal_evidence = [{
      record_id: 'observation-a',
      kind: 'observation',
      author: 'agent_000001',
      tick: 4,
      summary: 'Observed fungal grove at (4, 5)',
      content: '{"terrain":"FUNGAL_GROVE"}',
      related_artifacts: [],
    }]
    snapshot.artifacts.ids = ['artifact-a']
    snapshot.artifacts.name = ['Fungal lattice']
    snapshot.artifacts.creator = ['agent_000002']
    snapshot.artifacts.causal_parents = [['insight-a']]
    snapshot.artifacts.program = ['']
    snapshot.artifacts.program_id = ['']
    const messages = [1, 2].map((tick) => ({
      tick,
      kind: 'message_delivered',
      payload: { sender: 'agent_000001', recipients: ['agent_000002'] },
    }))

    const elements = buildLineageElements(snapshot, messages)
    expect(elements.find(({ data }) => data.id === 'insight-a')?.data.label).toBe('A concise public finding.')
    expect(elements.some(({ data }) => data.source === 'agent_000001' && data.target === 'observation-a' && data.relation === 'observed')).toBe(true)
    expect(elements.some(({ data }) => data.source === 'agent_000001' && data.target === 'insight-a' && data.relation === 'authored')).toBe(true)
    expect(elements.some(({ data }) => data.source === 'agent_000002' && data.target === 'artifact-a' && data.relation === 'built')).toBe(true)
    expect(elements.find(({ data }) => data.source === 'agent_000001' && data.target === 'agent_000002' && data.relation === 'communicated')?.data.count).toBe(2)
  })
})
