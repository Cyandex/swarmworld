import type { HistoryFrame, Snapshot } from '../types'

export function mergeReplayFrames(
  current: HistoryFrame[],
  incoming: HistoryFrame[],
  maximum = 256,
): HistoryFrame[] {
  const byTick = new Map<number, HistoryFrame>()
  current.forEach((frame) => byTick.set(frame.snapshot.tick, frame))
  incoming.forEach((frame) => byTick.set(frame.snapshot.tick, frame))
  const ordered = [...byTick.values()].sort((a, b) => a.snapshot.tick - b.snapshot.tick)
  if (ordered.length <= maximum) return ordered
  if (maximum <= 1) return [ordered[ordered.length - 1]]
  const sampled: HistoryFrame[] = []
  for (let index = 0; index < maximum; index += 1) {
    const source = Math.round(index * (ordered.length - 1) / (maximum - 1))
    const frame = ordered[source]
    if (sampled[sampled.length - 1]?.snapshot.tick !== frame.snapshot.tick) sampled.push(frame)
  }
  return sampled
}

export function shouldAdvanceDemo(
  socketOpen: boolean,
  hasAuthoritativeSnapshot: boolean,
  paused: boolean,
): boolean {
  return !socketOpen && !hasAuthoritativeSnapshot && !paused
}

export function shouldResetLiveTimeline(
  hasAuthoritativeSnapshot: boolean,
  current: Snapshot,
  next: Snapshot,
): boolean {
  if (!hasAuthoritativeSnapshot) return true
  return current.seed !== next.seed || next.tick < current.tick
}
