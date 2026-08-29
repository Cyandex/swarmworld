export const TERRAIN_NAMES = [
  'Deep water',
  'Tidal shelf',
  'Meadow',
  'Fungal grove',
  'Chitin garden',
  'Cellulose field',
  'Mineral spring',
  'Foundry',
  'Test field',
]

export const RESOURCE_NAMES = [
  'None',
  'Kelp',
  'Shell',
  'Fungus',
  'Chitin',
  'Cellulose',
  'Mineral',
  'Water',
  'Catalyst',
]

export const STATION_NAMES = [
  'None',
  'Fermenter',
  'Washer',
  'Press',
  'Aligner',
  'Tester',
  'Archive',
]

export const ACTION_NAMES = [
  'Waiting',
  'Moving',
  'Inspecting',
  'Harvesting',
  'Depositing',
  'Operating',
  'Testing',
  'Proposing recipe',
  'Building',
  'Repairing',
  'Dismantling',
  'Communicating',
  'Publishing',
  'Depositing insight',
  'Writing program',
  'Forking program',
  'Claiming task',
  'Teaching',
  'Trading',
  'Combining designs',
  'Metabolizing',
]

export const ARTIFACT_NAMES = [
  'None',
  'Material system',
]

export const AGENT_COLORS = [
  '#ff765f', '#ffc857', '#68d5ff', '#ad8cff', '#71e0bd', '#9ee06f',
  '#ff79c9', '#d99bff', '#60cbd0', '#e8e56a', '#ff9c62', '#8aa7ff',
]

export const FIELD_LABELS = {
  terrain: 'Living terrain',
  moisture: 'Moisture',
  nutrients: 'Nutrients',
  contamination: 'Contamination',
  temperature: 'Temperature',
} as const
