// Name tables mirror web/src/constants.ts (unmodified source); colors are this
// client's own top-down palette. Scenario worlds may override names/colors via
// snapshot.scenario.rendering in a later milestone.

export const TERRAIN_NAMES = [
  'Deep water', 'Tidal shelf', 'Meadow', 'Fungal grove', 'Chitin garden',
  'Cellulose field', 'Mineral spring', 'Foundry', 'Test field',
]

export const TERRAIN_COLORS = [
  '#10304a', '#2a5d6e', '#3f7d44', '#6b4f7e', '#8a6b3d',
  '#7d9c4a', '#4a8f9c', '#6e6e78', '#9c8f4a',
]

export const RESOURCE_NAMES = [
  'None', 'Kelp', 'Shell', 'Fungus', 'Chitin',
  'Cellulose', 'Mineral', 'Water', 'Catalyst',
]

export const RESOURCE_COLORS = [
  'transparent', '#37b06b', '#e8e0cf', '#c77dff', '#d9973b',
  '#b6d94b', '#7dd3e8', '#4aa3ff', '#ff5d8f',
]

export const STATION_NAMES = [
  'None', 'Fermenter', 'Washer', 'Press', 'Aligner', 'Tester', 'Archive',
]

export const ACTION_NAMES = [
  'Waiting', 'Moving', 'Inspecting', 'Harvesting', 'Depositing', 'Operating',
  'Testing', 'Proposing recipe', 'Building', 'Repairing', 'Dismantling',
  'Communicating', 'Publishing', 'Depositing insight', 'Writing program',
  'Forking program', 'Claiming task', 'Teaching', 'Trading',
  'Combining designs', 'Metabolizing',
]

export const AGENT_COLORS = [
  '#ff765f', '#ffc857', '#68d5ff', '#ad8cff', '#71e0bd', '#9ee06f',
  '#ff79c9', '#d99bff', '#60cbd0', '#e8e56a', '#ff9c62', '#8aa7ff',
]

export const FIELD_OVERLAYS = [
  { key: null, label: 'no field overlay' },
  { key: 'moisture', label: 'moisture', color: [74, 163, 255] },
  { key: 'contamination', label: 'contamination', color: [255, 93, 95] },
  { key: 'nutrients', label: 'nutrients', color: [113, 224, 133] },
  { key: 'temperature', label: 'temperature', color: [255, 200, 87] },
]

export const EVENT_LABELS = {
  agents_moved: null, // too chatty to display
  salient_discovery: 'discovery',
  action_rejected: 'rejected',
  action_result: 'result',
  resource_harvested: 'harvested',
  resource_deposited: 'deposited',
  resource_traded: 'trade',
  sample_inspected: 'inspected',
  recipe_proposed: 'recipe proposed',
  design_combined: 'designs combined',
  microbatch_fabricated: 'fabricated',
  material_tested: 'tested',
  artifact_built: 'artifact built',
  artifact_repaired: 'artifact repaired',
  artifact_dismantled: 'artifact dismantled',
  artifact_milestone: 'artifact milestone',
  artifact_program_installed: 'program installed',
  insight_deposited: 'insight',
  message_delivered: 'message',
  knowledge_taught: 'taught',
  task_claimed: 'task claimed',
  request_fulfilled: 'request fulfilled',
  environmental_disturbance: 'DISTURBANCE',
  research_milestone: 'milestone',
  research_mission_completed: 'MISSION COMPLETE',
  agent_died: 'agent died',
  agent_respawned: 'agent respawned',
  agent_deliberated: 'thought',
  model_error: 'model error',
  human_intervention: 'you acted',
}
