import { LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { init, use, type ECharts } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { useEffect, useRef, useState } from 'react'
import { ACTION_NAMES } from '../constants'
import type { DynamicsHistory, DynamicsPoint } from '../types'

use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

type ChartView = 'overview' | 'society' | 'mobility' | 'actions'

const SOCIETY_SERIES: Array<{ key: keyof DynamicsPoint; name: string; color: string; dash?: boolean }> = [
  { key: 'mean_energy', name: 'Energy', color: '#70d7bc' },
  { key: 'research_score', name: 'Research', color: '#8fb5ff', dash: true },
  { key: 'regional_crowding', name: 'Regional crowding', color: '#ef9f68' },
  { key: 'communication_rate', name: 'Communication', color: '#df7ec4' },
  { key: 'specialization', name: 'Specialization', color: '#c5a9ff' },
  { key: 'behavioral_diversity', name: 'Diversity', color: '#b9dc6e' },
]

const ACTION_COLORS = [
  '#6d837b', '#8fb5ff', '#efc968', '#8dd5be', '#cf9de8', '#e49c68', '#82c5df',
  '#c4da75', '#f08a75', '#9fb6d8', '#be8d7b', '#66c7ae', '#e6b966', '#b294dc',
  '#6eb7ca', '#a9d789', '#f29daf', '#d7c482', '#82a6ed', '#7ed5b7',
]

function displayActionName(name: string, index: number) {
  return ACTION_NAMES[index] ?? name.toLowerCase().replaceAll('_', ' ')
}

function baseOption(ticks: number[]) {
  return {
    animation: false,
    backgroundColor: 'transparent',
    color: ACTION_COLORS,
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#10201d',
      borderColor: '#34594e',
      textStyle: { color: '#e6efe8', fontFamily: 'DM Mono', fontSize: 9 },
    },
    xAxis: {
      type: 'category', data: ticks, boundaryGap: false,
      axisLine: { lineStyle: { color: '#2e4943' } }, axisTick: { show: false },
      axisLabel: { color: '#78938a', fontSize: 8, formatter: (value: string, index: number) => index % Math.max(1, Math.floor(ticks.length / 5)) === 0 ? value : '' },
    },
  }
}

function createOption(history: DynamicsHistory, view: ChartView) {
  const points = history.points
  const ticks = points.map(({ tick }) => tick)
  const common = baseOption(ticks)
  if (view === 'overview') {
    return {
      ...common,
      grid: { left: 35, right: 34, top: 25, bottom: 21 },
      legend: { top: 1, textStyle: { color: '#78938a', fontSize: 8 }, itemWidth: 12, itemHeight: 2 },
      yAxis: [
        { type: 'value', axisLabel: { color: '#78938a', fontSize: 8 }, splitLine: { lineStyle: { color: '#172b27' } } },
        { type: 'value', minInterval: 1, axisLabel: { color: '#78938a', fontSize: 8 }, splitLine: { show: false } },
      ],
      series: [
        {
          name: 'Artifact utility', type: 'line', smooth: true, showSymbol: false,
          data: points.map(({ artifact_utility }) => artifact_utility),
          lineStyle: { width: 2, color: '#f0c768' }, areaStyle: { color: 'rgba(240,199,104,.08)' },
        },
        {
          name: 'Artifact count', type: 'line', step: 'end', showSymbol: false, yAxisIndex: 1,
          data: points.map(({ artifact_count }) => artifact_count),
          lineStyle: { width: 1.5, color: '#7ed5b7', type: 'dashed' },
        },
      ],
    }
  }
  if (view === 'society') {
    return {
      ...common,
      grid: { left: 35, right: 13, top: 35, bottom: 21 },
      legend: { type: 'scroll', top: 0, textStyle: { color: '#78938a', fontSize: 7 }, itemWidth: 10, itemHeight: 2, pageTextStyle: { color: '#78938a', fontSize: 7 } },
      yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#78938a', fontSize: 8 }, splitLine: { lineStyle: { color: '#172b27' } } },
      series: SOCIETY_SERIES.map(({ key, name, color, dash }) => ({
        name, type: 'line', smooth: key !== 'research_score', step: key === 'research_score' ? 'end' : undefined, showSymbol: false,
        data: points.map((point) => Number(point[key] ?? 0)),
        lineStyle: { width: 1.45, color, type: dash ? 'dashed' : 'solid' },
      })),
    }
  }
  if (view === 'mobility') {
    return {
      ...common,
      grid: { left: 40, right: 38, top: 35, bottom: 21 },
      legend: { top: 0, textStyle: { color: '#78938a', fontSize: 7 }, itemWidth: 10, itemHeight: 2 },
      yAxis: [
        { type: 'value', min: 0, axisLabel: { color: '#78938a', fontSize: 8 }, splitLine: { lineStyle: { color: '#172b27' } } },
        { type: 'value', min: 0, max: 1, axisLabel: { color: '#78938a', fontSize: 8 }, splitLine: { show: false } },
      ],
      series: [
        {
          name: 'Mean travel', type: 'line', smooth: true, showSymbol: false,
          data: points.map(({ mean_distance_traveled }) => mean_distance_traveled ?? 0),
          lineStyle: { width: 1.8, color: '#8fb5ff' },
        },
        {
          name: 'Cells visited', type: 'line', smooth: true, showSymbol: false,
          data: points.map(({ mean_distinct_cells_visited }) => mean_distinct_cells_visited ?? 0),
          lineStyle: { width: 1.6, color: '#7ed5b7', type: 'dashed' },
        },
        {
          name: 'Regional crowding', type: 'line', smooth: true, showSymbol: false, yAxisIndex: 1,
          data: points.map(({ regional_crowding, spatial_concentration }) => regional_crowding ?? spatial_concentration),
          lineStyle: { width: 1.5, color: '#ef9f68' },
        },
        {
          name: 'Moving now', type: 'line', smooth: true, showSymbol: false, yAxisIndex: 1,
          data: points.map(({ moving_agent_fraction }) => moving_agent_fraction ?? 0),
          lineStyle: { width: 1.2, color: '#df7ec4' },
        },
      ],
    }
  }

  const activeActions = history.action_names.flatMap((name, actionIndex) => (
    points.some((point) => (point.action_distribution[actionIndex] ?? 0) > 0.00001)
      ? [{ name, actionIndex }]
      : []
  ))
  return {
    ...common,
    grid: { left: 35, right: 13, top: 35, bottom: 21 },
    legend: { type: 'scroll', top: 0, textStyle: { color: '#78938a', fontSize: 7 }, itemWidth: 9, itemHeight: 5, pageTextStyle: { color: '#78938a', fontSize: 7 } },
    yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#78938a', fontSize: 8, formatter: (value: number) => `${Math.round(value * 100)}%` }, splitLine: { lineStyle: { color: '#172b27' } } },
    series: activeActions.map(({ name, actionIndex }, order) => ({
      name: displayActionName(name, actionIndex), type: 'line', stack: 'action-share', showSymbol: false, symbol: 'none',
      data: points.map((point) => point.action_distribution[actionIndex] ?? 0),
      lineStyle: { width: 0.6, color: ACTION_COLORS[order % ACTION_COLORS.length] },
      areaStyle: { opacity: 0.58, color: ACTION_COLORS[order % ACTION_COLORS.length] },
    })),
  }
}

export function MetricChart({ history }: { history: DynamicsHistory }) {
  const element = useRef<HTMLDivElement>(null)
  const chartRef = useRef<ECharts | null>(null)
  const historyRef = useRef(history)
  const viewRef = useRef<ChartView>('overview')
  const [view, setView] = useState<ChartView>('overview')
  historyRef.current = history
  viewRef.current = view

  useEffect(() => {
    if (!element.current) return
    let animationFrame = 0
    const observer = new ResizeObserver(() => chartRef.current?.resize())
    observer.observe(element.current)

    const initialize = () => {
      if (!element.current || element.current.clientWidth === 0 || element.current.clientHeight === 0) {
        animationFrame = window.requestAnimationFrame(initialize)
        return
      }
      chartRef.current = init(element.current, undefined, { renderer: 'canvas' })
      chartRef.current.setOption(createOption(historyRef.current, viewRef.current))
    }
    animationFrame = window.requestAnimationFrame(initialize)

    return () => {
      window.cancelAnimationFrame(animationFrame)
      observer.disconnect()
      chartRef.current?.dispose()
      chartRef.current = null
    }
  }, [])

  useEffect(() => {
    chartRef.current?.setOption(createOption(history, view), { notMerge: true, lazyUpdate: true })
  }, [history, view])

  const firstTick = history.points[0]?.tick ?? 0
  const lastTick = history.points[history.points.length - 1]?.tick ?? firstTick
  return (
    <div className="metric-chart-shell">
      <div className="metric-chart-tabs" aria-label="Society dynamics metric group">
        {(['overview', 'society', 'mobility', 'actions'] as ChartView[]).map((name) => <button key={name} className={view === name ? 'active' : ''} onClick={() => setView(name)}>{name}</button>)}
        <span>{history.points.length} samples · T{firstTick}–T{lastTick}</span>
      </div>
      <div ref={element} className="metric-chart" aria-label={`Full-run society dynamics from tick ${firstTick} to ${lastTick}`} />
    </div>
  )
}
