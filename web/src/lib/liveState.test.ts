import { describe, expect, it } from 'vitest'
import { createDemoSnapshot } from './demo'
import { mergeReplayFrames, shouldAdvanceDemo, shouldResetLiveTimeline } from './liveState'

describe('live state recovery', () => {
  it('never demo-evolves an authoritative frame during a disconnect', () => {
    expect(shouldAdvanceDemo(false, true, false)).toBe(false)
    expect(shouldAdvanceDemo(false, false, false)).toBe(true)
    expect(shouldAdvanceDemo(true, false, false)).toBe(false)
  })

  it('resets demo history on the first server frame and preserves a continuing run', () => {
    const current = createDemoSnapshot(17, 100)
    const continued = createDemoSnapshot(17, 101)

    expect(shouldResetLiveTimeline(false, current, continued)).toBe(true)
    expect(shouldResetLiveTimeline(true, current, continued)).toBe(false)
  })

  it('recognizes a server restart by seed change or tick regression', () => {
    const current = createDemoSnapshot(17, 100)
    expect(shouldResetLiveTimeline(true, current, createDemoSnapshot(18, 101))).toBe(true)
    expect(shouldResetLiveTimeline(true, current, createDemoSnapshot(17, 2))).toBe(true)
  })

  it('rehydrates out-of-order replay checkpoints without duplicating ticks', () => {
    const frame = (tick: number) => ({ snapshot: createDemoSnapshot(17, tick), events: [] })
    const restored = mergeReplayFrames(
      [frame(96)],
      [frame(0), frame(48), frame(96), frame(24), frame(72)],
    )

    expect(restored.map(({ snapshot }) => snapshot.tick)).toEqual([0, 24, 48, 72, 96])
  })

  it('bounds replay memory while preserving the beginning and current state', () => {
    const frames = Array.from({ length: 20 }, (_, tick) => ({
      snapshot: createDemoSnapshot(17, tick),
      events: [],
    }))
    const compacted = mergeReplayFrames([], frames, 5)

    expect(compacted).toHaveLength(5)
    expect(compacted[0].snapshot.tick).toBe(0)
    expect(compacted[4].snapshot.tick).toBe(19)
  })
})
