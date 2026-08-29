# BioFoundry Observatory

The Observatory is the browser-based Three.js renderer and scientific interface for
BioFoundry World. The Python simulator remains authoritative. The browser consumes
protocol-v1 snapshots and events, interpolates them for presentation, and can submit
only the bounded commands documented in `../docs/PROTOCOL.md`.

## Run locally

In one terminal, from the repository root:

```bash
conda activate PyTorch
biofoundry serve --config configs/demo.yaml --record runs/web-live.jsonl
```

In a second terminal:

```bash
cd web
npm ci
npm run dev
```

Open the local URL printed by Vite, normally `http://127.0.0.1:5173`.

If no Python server is available, the client enters an explicitly labeled,
deterministic demonstration state. The client retries failed handshakes, detects silent
stalls, and reconnects with bounded backoff. The Python server sends heartbeats while
LLM requests are in flight. `npm run dev` also supervises and restarts Vite after an
unexpected exit; Ctrl-C remains an intentional stop.
The demonstration is a renderer preview, not scientific output.

To connect to a server at another address, copy `.env.example` to `.env.local` and
change `VITE_BIOFOUNDRY_WS`.

## Production build

```bash
npm run build
npm run preview
```

The build output is written to `dist/`. It is a static client, but it must be able to
reach the configured BioFoundry WebSocket endpoint at runtime.

## Interface

- Drag to orbit, right-drag to pan, and scroll to zoom.
- The living view uses a continuous seeded heightfield, clipped animated coastal
  water, surface detail, biome-specific organisms and minerals, atmospheric lighting,
  and shadows. These presentation effects remain aligned to the authoritative grid.
- Select terrain, moisture, nutrient, contamination, or temperature views.
- Toggle resources, recent local exchanges, and selected-agent trajectories.
- Click an agent to inspect it and issue a one-tick bounded action.
- Click any world cell to inspect its terrain, station, resource mass, environmental
  fields, agents, and artifacts. The world-object index provides a searchable path to
  objects that are occluded or outside the current view.
- Click an artifact to inspect its identity, creator, geometry, current and peak
  services, causal parents, and installed tick program. Co-located artifacts are
  visually fanned out but retain their authoritative grid coordinate.
- Use Activity for the event stream, Lineage for the interactive provenance graph,
  and Report for a deterministic situation summary. Clicking a lineage node opens
  its recorded evidence, author, tick, parents, and measured artifact state. Program
  nodes expose immutable IDs and instructions. Relations are typed rather than
  visually interchangeable: agents **observe**, **test**, **author**, and **build**;
  programs are **forked** and **installed**; publications retain their causal parents.
  Repeated recent messages are aggregated by directed agent pair.
- Use **Paper snapshot** over the world or **Snapshot** beside the observer tabs to
  download both a true-vector SVG and a matching 2× PNG. These exports are rebuilt
  from authoritative state on a white background: the world includes its selected
  field, resources, laboratories, agents, artifacts, labels, and legend; Inspect,
  Activity, Lineage, and Report each use a purpose-built publication layout. The
  lineage export places actors, evidence, insights, programs, and artifacts in fixed
  semantic lanes and puts recent communication in a separate summary, keeping social
  traffic from obscuring the scientific causal graph.
- In Lineage, **Lanes** downloads that semantic scientific-flow figure and **Graph**
  downloads a second white-background SVG/PNG that preserves the current interactive
  force-directed node positions. The graph export is rebuilt as editable vector
  geometry—never captured from the dark canvas—and marks its layout as presentation-only.
- The Society dynamics dock hydrates the full authoritative run history whenever a
  viewer connects. **Overview** shows utility and artifact count, **Society** shows
  normalized energy, research, concentration, communication, specialization, and
  diversity, and **Actions** shows the rolling action-share composition.
- Drag the bottom timeline backward to inspect recent client-side frames; press
  **LIVE** to return to the authoritative head state.

Human interventions are disabled while viewing a historical client frame. The world
replay slider remains a recent, browser-local frame buffer; unlike that slider, the
dynamics charts recover the full run after a browser restart. The browser never
mutates state optimistically.

Snapshot filenames contain the seed, tick, and panel, so exports from different
conditions can be collected without manual renaming. The SVGs contain only standard
vector primitives and text—no HTML `foreignObject` or renderer canvas—making them
editable in Illustrator, Inkscape, and common journal workflows.

## Scaling behavior

Terrain, resources, and agents use Three.js instanced meshes. A large population
therefore does not create one React component or draw call per agent. Protocol v1
still transmits full presentation snapshots; future delta frames and level-of-detail
aggregation can be added without changing simulation semantics.
