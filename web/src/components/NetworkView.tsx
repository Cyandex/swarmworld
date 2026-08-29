import cytoscape, { type Core } from 'cytoscape'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  buildLineageElements,
  type LineageGraphLayout,
  type LineageNodeDetail,
} from '../lib/lineage'
import type { Snapshot, WorldEvent } from '../types'

interface NetworkViewProps {
  snapshot: Snapshot
  events: WorldEvent[]
  selectedNodeId?: string | null
  onSelectNode?: (detail: LineageNodeDetail | null) => void
  onLayoutChange?: (layout: LineageGraphLayout) => void
}

export function NetworkView({ snapshot, events, selectedNodeId, onSelectNode, onLayoutChange }: NetworkViewProps) {
  const element = useRef<HTMLDivElement>(null)
  const graphRef = useRef<Core | null>(null)
  const onSelectNodeRef = useRef(onSelectNode)
  const onLayoutChangeRef = useRef(onLayoutChange)
  const [error, setError] = useState<string | null>(null)
  const elements = useMemo(() => buildLineageElements(snapshot, events.slice(-120)), [events, snapshot])
  const topologySignature = JSON.stringify(elements.map(({ data }) => ({
    id: data.id,
    label: data.label,
    kind: data.kind,
    source: data.source,
    target: data.target,
    relation: data.relation,
    count: data.count,
  })))

  useEffect(() => {
    onSelectNodeRef.current = onSelectNode
  }, [onSelectNode])

  useEffect(() => {
    onLayoutChangeRef.current = onLayoutChange
  }, [onLayoutChange])

  useEffect(() => {
    if (!element.current) return
    setError(null)
    try {
      const graph = cytoscape({
        container: element.current,
        elements,
        style: [
          { selector: 'node', style: { label: 'data(label)', color: '#b9ccc4', 'font-size': 8, 'text-wrap': 'ellipsis', 'text-max-width': '90px', 'text-valign': 'bottom', 'text-margin-y': 6, width: 20, height: 20, 'background-color': '#e5bd63', 'border-width': 2, 'border-color': '#1f3b34' } },
          { selector: 'node[kind = "evidence"]', style: { shape: 'round-rectangle', 'background-color': '#9e8351', width: 17, height: 17, 'font-size': 7 } },
          { selector: 'node[kind = "artifact"]', style: { shape: 'diamond', 'background-color': '#7ed5b7', width: 25, height: 25 } },
          { selector: 'node[kind = "agent"]', style: { shape: 'ellipse', 'background-color': '#82b9df', width: 14, height: 14, 'font-size': 7 } },
          { selector: 'node[kind = "program"]', style: { shape: 'hexagon', 'background-color': '#b78ad9', width: 21, height: 21, 'font-size': 7 } },
          { selector: 'node:selected', style: { 'border-width': 4, 'border-color': '#fff0a0', 'overlay-color': '#fff0a0', 'overlay-opacity': 0.08, 'overlay-padding': 7 } },
          { selector: 'edge', style: { width: 1.1, 'curve-style': 'bezier', 'line-color': '#536f65', 'target-arrow-color': '#536f65', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.65, opacity: 0.65 } },
          { selector: 'edge[relation = "authored"]', style: { 'line-style': 'dotted', 'line-color': '#6da2c7', 'target-arrow-color': '#6da2c7', opacity: 0.72 } },
          { selector: 'edge[relation = "observed"]', style: { 'line-color': '#4e90b7', 'target-arrow-color': '#4e90b7', opacity: 0.78 } },
          { selector: 'edge[relation = "tested"]', style: { 'line-color': '#c49a3c', 'target-arrow-color': '#c49a3c', opacity: 0.82 } },
          { selector: 'edge[relation = "built"]', style: { width: 2.5, 'line-color': '#50b793', 'target-arrow-color': '#50b793', opacity: 0.9 } },
          { selector: 'edge[relation = "installed"]', style: { width: 2, 'line-color': '#a978c7', 'target-arrow-color': '#a978c7', opacity: 0.86 } },
          { selector: 'edge[relation = "forked"]', style: { width: 1.8, 'line-style': 'dashed', 'line-color': '#b78ad9', 'target-arrow-color': '#b78ad9', opacity: 0.86 } },
          { selector: 'edge[relation = "communicated"]', style: { label: 'data(countLabel)', 'font-size': 6, color: '#73cdb6', 'text-background-color': '#10251f', 'text-background-opacity': 0.8, 'text-background-padding': '2px', width: 'mapData(count, 1, 12, 0.8, 3.2)', 'line-style': 'dashed', 'line-color': '#5ebfa6', 'target-arrow-color': '#5ebfa6', opacity: 0.58 } },
        ],
        layout: { name: 'cose', animate: false, padding: 26, nodeRepulsion: () => 7000, idealEdgeLength: () => 78 },
        minZoom: 0.45,
        maxZoom: 2.4,
      })
      graphRef.current = graph
      graph.on('tap', 'node', (event) => {
        const detail = event.target.data('detail') as LineageNodeDetail | undefined
        onSelectNodeRef.current?.(detail ?? null)
      })
      graph.on('tap', (event) => {
        if (event.target === graph) onSelectNodeRef.current?.(null)
      })
      if (selectedNodeId) graph.getElementById(selectedNodeId).select()
      const layoutFrame = window.requestAnimationFrame(() => {
        const positions = Object.fromEntries(graph.nodes().map((node) => {
          const position = node.position()
          return [node.id(), { x: position.x, y: position.y }]
        }))
        if (Object.keys(positions).length) onLayoutChangeRef.current?.({ positions })
      })
      let resizeFrame = 0
      const resizeObserver = new ResizeObserver(() => {
        window.cancelAnimationFrame(resizeFrame)
        resizeFrame = window.requestAnimationFrame(() => {
          graph.resize()
          graph.fit(graph.elements(), 26)
        })
      })
      resizeObserver.observe(element.current)
      return () => {
        resizeObserver.disconnect()
        window.cancelAnimationFrame(layoutFrame)
        window.cancelAnimationFrame(resizeFrame)
        graphRef.current = null
        graph.destroy()
      }
    } catch (reason) {
      console.error('Unable to render lineage graph', reason)
      setError(reason instanceof Error ? reason.message : 'Unknown graph error')
    }
  // The serialized graph prevents teardown/re-layout on unrelated world ticks.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topologySignature])

  useEffect(() => {
    const graph = graphRef.current
    if (!graph) return
    elements.forEach(({ data }) => {
      if (data.source || data.target) return
      const node = graph.getElementById(data.id)
      if (node.empty()) return
      node.data('detail', data.detail)
      node.data('label', data.label)
    })
  }, [elements, topologySignature])

  useEffect(() => {
    const graph = graphRef.current
    if (!graph) return
    graph.nodes().unselect()
    if (selectedNodeId) graph.getElementById(selectedNodeId).select()
  }, [selectedNodeId, topologySignature])

  if (error) {
    return <div className="network-view network-error" role="alert">Lineage rendering paused: {error}</div>
  }
  return <div ref={element} className="network-view" aria-label="Interactive idea, artifact, and communication lineage network. Select a node to inspect its contents." />
}
