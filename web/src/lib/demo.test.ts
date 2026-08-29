import { describe, expect, it } from 'vitest'
import { createDemoSnapshot, evolveDemo } from './demo'

describe('deterministic renderer demonstration', () => {
  it('creates protocol-shaped, deterministic world state', () => {
    const first = createDemoSnapshot(17, 10)
    const second = createDemoSnapshot(17, 10)

    expect(first).toEqual(second)
    expect(first.protocol).toBe(1)
    expect(first.world.terrain).toHaveLength(first.world.width * first.world.height)
    expect(first.agents.ids).toHaveLength(first.agents.display_count)
    expect(first.artifacts.ids).toHaveLength(first.artifacts.display_count)
  })

  it('advances without mutating the prior snapshot', () => {
    const before = createDemoSnapshot(17, 10)
    const position = before.agents.x[1]
    const { snapshot: after } = evolveDemo(before)

    expect(after.tick).toBe(11)
    expect(before.tick).toBe(10)
    expect(before.agents.x[1]).toBe(position)
    expect(after.world.terrain).toEqual(before.world.terrain)
  })
})
