export type NumericArray = number[]

export interface ScenarioCatalogRecord {
  slot: string
  id: string
  name?: string
  color?: string
  height?: number
  walkable?: boolean
  [key: string]: unknown
}

export interface ScenarioDescriptor {
  id: string
  name: string
  version: string
  package_hash: string
  description?: string
  catalogs: Record<string, ScenarioCatalogRecord[]>
  fields?: Array<{ id: string; name?: string; [key: string]: unknown }>
  services?: Array<Record<string, unknown>>
  mission?: Record<string, unknown>
}

export interface WorldState {
  width: number
  height: number
  terrain: NumericArray
  resource_kind: NumericArray
  resource_mass: NumericArray
  stations: NumericArray
  temperature: NumericArray
  moisture: NumericArray
  nutrients: NumericArray
  contamination: NumericArray
  solar: NumericArray
  fields?: Record<string, NumericArray>
  scenario?: ScenarioDescriptor
  generator_manifest?: Record<string, unknown>
  flux_ledger?: Record<string, number>
}

export interface AgentState {
  count: number
  display_count: number
  ids: string[]
  x: NumericArray
  y: NumericArray
  energy: NumericArray
  active?: boolean[]
  generation?: NumericArray
  visor: NumericArray
  inventory_total: NumericArray
  last_action: NumericArray
  distance_traveled?: NumericArray
  distinct_cells_visited?: NumericArray
}

export interface ArtifactState {
  count: number
  display_count: number
  ids: string[]
  kind: NumericArray
  x: NumericArray
  y: NumericArray
  health: NumericArray
  maturity: NumericArray
  performance: NumericArray
  peak_performance: NumericArray
  lifetime_peak_performance?: NumericArray
  lifetime_peak_tick?: NumericArray
  lifetime_peak_program_id?: string[]
  storage: NumericArray
  open_fraction: NumericArray
  program: string[]
  program_id?: string[]
  retired?: boolean[]
  name: string[]
  claimed_function: string[]
  architecture: string[]
  bio_inspiration: string[][]
  geometry: Array<Record<string, number>>
  creator?: string[]
  created_tick?: NumericArray
  causal_parents?: string[][]
  services: Record<string, NumericArray>
  peak_services: Record<string, NumericArray>
  lifetime_peak_services?: Record<string, NumericArray>
}

export interface ArchiveRecord {
  id?: string
  record_id?: string
  author?: string
  title?: string
  content?: string
  kind?: string
  tick?: number
  [key: string]: unknown
}

export interface InsightRecord {
  id?: string
  record_id?: string
  author?: string
  title?: string
  claim?: string
  content?: string
  kind?: string
  x?: number
  y?: number
  tick?: number
  causal_parents?: string[]
  [key: string]: unknown
}

export interface CausalEvidenceRecord {
  record_id: string
  kind: string
  author?: string
  tick?: number
  summary: string
  content?: string
  causal_parents?: string[]
  related_artifacts?: string[]
}

export interface WorldMetrics {
  artifact_score: number
  archive_entries: number
  deposited_insights: number
  research_score?: number
  research_completed?: boolean
}

export interface ResearchScorecard {
  mission: string
  score: number
  completed: boolean
  outcome_success: boolean
  emergent_success: boolean
  milestones: Record<string, boolean>
  counts: Record<string, number>
  best_material_utility: number
  best_artifact_performance: number
  best_behavioral_novelty: number
  best_independent_recipe_utility: number | null
  best_combined_recipe_utility: number | null
  composition_gain: number | null
  composition_synergy?: number | null
  inventions: Array<{
    artifact_id: string
    name: string
    architecture: string
    claimed_function: string
    bio_inspiration: string[]
    performance: number
    behavioral_novelty: number
    services: Record<string, number>
  }>
  depot: Record<string, number>
}

export interface Snapshot {
  protocol: number
  tick: number
  seed: number
  scenario?: ScenarioDescriptor
  world: WorldState
  agents: AgentState
  artifacts: ArtifactState
  archive: ArchiveRecord[]
  insights: InsightRecord[]
  causal_evidence?: CausalEvidenceRecord[]
  metrics: WorldMetrics
  research?: ResearchScorecard
  program_lineage?: ProgramLineageEdge[]
  program_catalog?: ProgramCatalogRecord[]
}

export interface ProgramLineageEdge {
  parent_program_id: string
  child_program_id: string
  tick: number
  artifact_id: string
  author: string
  instruction_diff: Array<Record<string, unknown>>
}

export interface ProgramCatalogRecord {
  program_id: string
  name: string
  instructions: Array<Record<string, unknown>>
  authors: string[]
  first_tick: number
  installations: Array<Record<string, unknown>>
}

export interface WorldEvent {
  tick: number
  kind: string
  payload: Record<string, unknown>
}

export interface ControlState {
  paused: boolean
  speed_multiplier: number
  step_budget: number
  model_requests: number
  model_errors: number
  provider_attempts?: number
  provider_outage?: boolean
  llm_enabled: boolean
  model: string
  playback_mode?: boolean
  playback_complete?: boolean
  max_tick?: number
  trace_name?: string
}

export type ConnectionMode = 'connecting' | 'reconnecting' | 'live' | 'playback' | 'demo' | 'complete'

export interface HistoryFrame {
  snapshot: Snapshot
  events: WorldEvent[]
}

export interface DynamicsPoint {
  tick: number
  artifact_utility: number
  mean_energy: number
  spatial_concentration: number
  regional_crowding?: number
  mean_distance_traveled: number
  mean_distinct_cells_visited: number
  moving_agent_fraction: number
  research_score: number
  artifact_count: number
  communication_rate: number
  specialization: number
  behavioral_diversity: number
  action_distribution: number[]
}

export interface DynamicsHistory {
  version: number
  sample_interval: number
  role_window: number
  action_names: string[]
  points: DynamicsPoint[]
}

export type FieldMode = string

export interface ManualAction {
  verb: number
  direction?: number
  resource?: number
  artifact?: number
  target_x?: number
  target_y?: number
  amount?: number
}

export interface ClientCommand {
  command: 'toggle_pause' | 'set_paused' | 'step' | 'set_speed' | 'manual_action' | 'seek' | 'play_from_start'
  paused?: boolean
  speed_multiplier?: number
  tick?: number
  agent?: string
  action?: ManualAction
}
