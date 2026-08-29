import { ACTION_NAMES } from '../constants'
import type { DynamicsHistory, DynamicsPoint, Snapshot } from '../types'

export const FALLBACK_ACTION_NAMES = ACTION_NAMES.map((name) => name.toUpperCase().replaceAll(' ', '_'))

function normalizedEntropy(distribution: number[]) {
  if (distribution.length < 2) return 0
  const entropy = distribution.reduce((sum, probability) => (
    probability > 0 ? sum - probability * Math.log(probability) : sum
  ), 0)
  return entropy / Math.log(distribution.length)
}

export function dynamicsPointFromSnapshot(snapshot: Snapshot): DynamicsPoint {
  const population = Math.max(1, snapshot.agents.last_action.length)
  const counts = new Array(FALLBACK_ACTION_NAMES.length).fill(0) as number[]
  snapshot.agents.last_action.forEach((action) => {
    if (action >= 0 && action < counts.length) counts[action] += 1
  })
  const observed = Math.max(1, snapshot.agents.last_action.length)
  const actionDistribution = counts.map((count) => count / observed)
  const cellCounts = new Map<string, number>()
  snapshot.agents.x.forEach((x, index) => {
    const key = `${x},${snapshot.agents.y[index]}`
    cellCounts.set(key, (cellCounts.get(key) ?? 0) + 1)
  })
  const hhi = [...cellCounts.values()].reduce((sum, count) => sum + (count / population) ** 2, 0)
  const baseline = 1 / population
  const spatialConcentration = population <= 1 ? 1 : Math.max(0, Math.min(1, (hhi - baseline) / (1 - baseline)))
  const regionalCounts = new Map<number, number>()
  snapshot.agents.x.forEach((x, index) => {
    const binX = Math.min(7, Math.max(0, Math.floor(x * 8 / Math.max(1, snapshot.world.width))))
    const binY = Math.min(5, Math.max(0, Math.floor(snapshot.agents.y[index] * 6 / Math.max(1, snapshot.world.height))))
    const key = binY * 8 + binX
    regionalCounts.set(key, (regionalCounts.get(key) ?? 0) + 1)
  })
  const regionalCrowding = population <= 1 ? 1 : (
    [...regionalCounts.values()].reduce((sum, count) => sum + count * (count - 1), 0)
    / (population * (population - 1))
  )

  return {
    tick: snapshot.tick,
    artifact_utility: snapshot.metrics.artifact_score,
    mean_energy: snapshot.agents.energy.reduce((sum, value) => sum + value, 0) / observed,
    spatial_concentration: spatialConcentration,
    regional_crowding: regionalCrowding,
    mean_distance_traveled: (
      snapshot.agents.distance_traveled?.reduce((sum, value) => sum + value, 0) ?? 0
    ) / observed,
    mean_distinct_cells_visited: (
      snapshot.agents.distinct_cells_visited?.reduce((sum, value) => sum + value, 0) ?? observed
    ) / observed,
    moving_agent_fraction: snapshot.agents.last_action.filter((action) => action === 1).length / observed,
    research_score: snapshot.research?.score ?? snapshot.metrics.research_score ?? 0,
    artifact_count: snapshot.artifacts.count,
    communication_rate: [11, 17, 18].reduce((sum, action) => sum + actionDistribution[action], 0),
    specialization: 0,
    behavioral_diversity: normalizedEntropy(actionDistribution),
    action_distribution: actionDistribution,
  }
}

export function initialDynamicsHistory(snapshot: Snapshot): DynamicsHistory {
  return {
    version: 2,
    sample_interval: 1,
    role_window: 1,
    action_names: FALLBACK_ACTION_NAMES,
    points: [dynamicsPointFromSnapshot(snapshot)],
  }
}

export function mergeDynamicsPoint(history: DynamicsHistory, point: DynamicsPoint): DynamicsHistory {
  const existing = history.points.findIndex(({ tick }) => tick === point.tick)
  if (existing >= 0) {
    const points = [...history.points]
    points[existing] = point
    return { ...history, points }
  }
  if (!history.points.length || history.points[history.points.length - 1].tick < point.tick) {
    return { ...history, points: [...history.points, point] }
  }
  return { ...history, points: [...history.points, point].sort((a, b) => a.tick - b.tick) }
}
