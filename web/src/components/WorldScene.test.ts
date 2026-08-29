import { describe, expect, it } from 'vitest'
import { createDemoSnapshot } from '../lib/demo'
import { artifactDisplayOffset } from './WorldScene'

describe('artifact presentation offsets', () => {
  it('separates co-located artifacts without moving unique artifacts', () => {
    const snapshot = createDemoSnapshot(17, 20)
    snapshot.artifacts.ids = ['artifact-a', 'artifact-b', 'artifact-c']
    snapshot.artifacts.x = [20, 20, 32]
    snapshot.artifacts.y = [13, 13, 23]

    const first = artifactDisplayOffset(snapshot, 0)
    const second = artifactDisplayOffset(snapshot, 1)

    expect(first).not.toEqual(second)
    expect(Math.hypot(...first)).toBeCloseTo(0.68)
    expect(Math.hypot(...second)).toBeCloseTo(0.68)
    expect(artifactDisplayOffset(snapshot, 2)).toEqual([0, 0])
  })
})
