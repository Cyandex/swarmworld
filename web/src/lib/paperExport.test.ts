import { describe, expect, it } from 'vitest'
import { createDemoSnapshot } from './demo'
import { buildLineageElements } from './lineage'
import {
  buildLineageNetworkPaperArtifact,
  buildPanelPaperArtifact,
  buildWorldPaperArtifact,
} from './paperExport'

describe('paper snapshot exports', () => {
  it('builds a white vector world state from simulation data', () => {
    const snapshot = createDemoSnapshot(17, 42)
    const artifact = buildWorldPaperArtifact(snapshot, 'terrain', null, null, null)

    expect(artifact.width).toBe(1600)
    expect(artifact.basename).toContain('world')
    expect(artifact.svg).toContain('fill="#ffffff"')
    expect(artifact.svg).toContain('SwarmWorld society state')
    expect(artifact.svg).toContain('authoritative state export')
    expect(artifact.svg).not.toContain('<foreignObject')
  })

  it.each(['inspect', 'activity', 'lineage', 'report'] as const)(
    'builds a white publication artifact for the %s panel',
    (tab) => {
      const snapshot = createDemoSnapshot(23, 71)
      const artifact = buildPanelPaperArtifact({
        tab,
        snapshot,
        events: [],
        selectedAgentIndex: -1,
        selectedArtifactIndex: -1,
        selectedTile: null,
        selectedLineageNode: null,
      })

      expect(artifact.basename).toContain(tab)
      expect(artifact.svg).toContain('fill="#ffffff"')
      expect(artifact.svg).toContain(`tick ${snapshot.tick}`)
      expect(artifact.svg).not.toContain('<foreignObject')
    },
  )

  it('separates typed scientific provenance from recent communication', () => {
    const snapshot = createDemoSnapshot(31, 88)
    snapshot.insights[0].author = 'agent_000000'
    snapshot.causal_evidence = [{
      record_id: 'observation-paper',
      kind: 'observation',
      author: 'agent_000000',
      tick: 3,
      summary: 'Observed cellulose field at (3, 4)',
    }]
    snapshot.insights[0].causal_parents = ['observation-paper']
    const artifact = buildPanelPaperArtifact({
      tab: 'lineage',
      snapshot,
      events: [{
        tick: 8,
        kind: 'message_delivered',
        payload: { sender: 'agent_000000', recipients: ['agent_000001'] },
      }],
      selectedAgentIndex: -1,
      selectedArtifactIndex: -1,
      selectedTile: null,
      selectedLineageNode: null,
    })

    expect(artifact.svg).toContain('SCIENTIFIC ACTORS')
    expect(artifact.svg).toContain('CAUSAL OBSERVATIONS &amp; TESTS')
    expect(artifact.svg).toContain('INVENTED ARTIFACTS')
    expect(artifact.svg).toContain('RECENT AGENT EXCHANGES')
    expect(artifact.svg).toContain('Observed cellulose field')
  })

  it('keeps unused draft programs out of the publication lineage', () => {
    const snapshot = createDemoSnapshot(37, 144)
    snapshot.artifacts.program_id = ['program-used', ...Array(snapshot.artifacts.count - 1).fill('')]
    snapshot.program_catalog = [
      { program_id: 'program-used', name: 'Installed adaptive loop', instructions: [], authors: ['agent_000000'], first_tick: 90, installations: [{}] },
      { program_id: 'program-unused', name: 'Unused private draft', instructions: [], authors: ['agent_000001'], first_tick: 91, installations: [] },
    ]
    const artifact = buildPanelPaperArtifact({
      tab: 'lineage', snapshot, events: [], selectedAgentIndex: -1,
      selectedArtifactIndex: -1, selectedTile: null, selectedLineageNode: null,
    })

    expect(artifact.svg).toContain('Installed adaptive loop')
    expect(artifact.svg).not.toContain('Unused private draft')
  })

  it('rebuilds a captured force layout as a white vector society graph', () => {
    const snapshot = createDemoSnapshot(41, 177)
    const events = [{
      tick: 170,
      kind: 'message_delivered',
      payload: { sender: 'agent_000000', recipients: ['agent_000001'] },
    }]
    const nodeIds = buildLineageElements(snapshot, events)
      .filter(({ data }) => !data.source && !data.target)
      .map(({ data }) => data.id)
    const positions = Object.fromEntries(nodeIds.map((id, index) => [
      id,
      { x: (index % 7) * 90, y: Math.floor(index / 7) * 72 },
    ]))
    const artifact = buildLineageNetworkPaperArtifact({
      tab: 'lineage', snapshot, events, selectedAgentIndex: -1,
      selectedArtifactIndex: -1, selectedTile: null, selectedLineageNode: null,
    }, { positions })

    expect(artifact.basename).toContain('lineage-graph')
    expect(artifact.svg).toContain('Force-directed scientific society graph')
    expect(artifact.svg).toContain('fill="#ffffff"')
    expect(artifact.svg).toContain('network-arrow-communicated')
    expect(artifact.svg).toContain('positions preserve the interactive force-directed layout')
    expect(artifact.svg).not.toContain('<foreignObject')
  })
})
