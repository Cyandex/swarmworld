import { describe, expect, it } from 'vitest'
import { createDemoSnapshot } from './demo'
import { dynamicsPointFromSnapshot, initialDynamicsHistory, mergeDynamicsPoint } from './dynamics'

describe('society dynamics history', () => {
  it('derives population-normalized legacy metrics from a snapshot', () => {
    const snapshot = createDemoSnapshot(17, 20)
    snapshot.agents.x = snapshot.agents.x.map((_, index) => index)
    snapshot.agents.y = snapshot.agents.y.map(() => 10)
    snapshot.agents.last_action = snapshot.agents.last_action.map(() => 11)

    const point = dynamicsPointFromSnapshot(snapshot)
    expect(point.tick).toBe(snapshot.tick)
    expect(point.spatial_concentration).toBeCloseTo(0)
    expect(point.communication_rate).toBeCloseTo(1)
    expect(point.action_distribution.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1)
  })

  it('replaces duplicate ticks and retains the complete ordered trajectory', () => {
    const snapshot = createDemoSnapshot(17, 20)
    const initial = initialDynamicsHistory(snapshot)
    const tick21 = { ...initial.points[0], tick: 21, artifact_utility: 1 }
    const corrected = { ...tick21, artifact_utility: 2 }
    const tick19 = { ...tick21, tick: 19 }

    const merged = mergeDynamicsPoint(mergeDynamicsPoint(mergeDynamicsPoint(initial, tick21), corrected), tick19)
    expect(merged.points.map(({ tick }) => tick)).toEqual([19, 20, 21])
    expect(merged.points.find(({ tick }) => tick === 21)?.artifact_utility).toBe(2)
  })
})
