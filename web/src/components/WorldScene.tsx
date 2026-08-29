import { Line, MapControls } from '@react-three/drei'
import { Canvas, type ThreeEvent, useFrame } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { AGENT_COLORS } from '../constants'
import type { FieldMode, Snapshot, WorldEvent } from '../types'

const TERRAIN_COLORS = [
  '#285b69', '#4c8980', '#6f9152', '#365f4d', '#8b7544',
  '#96a861', '#62a4a0', '#84796a', '#6f8279',
]
const TERRAIN_HEIGHTS = [-0.64, -0.15, 0.16, 0.46, 0.35, 0.31, 0.72, 0.25, 0.2]
const RESOURCE_COLORS = ['#000000', '#4fa873', '#e8d8ad', '#b783d2', '#d78343', '#c8c26c', '#b7d3d7', '#62bcd9', '#edb85d']

function scenarioCatalog(snapshot: Snapshot, kind: string) {
  return (snapshot.scenario ?? snapshot.world.scenario)?.catalogs?.[kind]
}

function terrainColor(snapshot: Snapshot, kind: number) {
  return String(scenarioCatalog(snapshot, 'terrains')?.[kind]?.color ?? TERRAIN_COLORS[kind] ?? '#66846c')
}

function terrainHeight(snapshot: Snapshot, kind: number) {
  return Number(scenarioCatalog(snapshot, 'terrains')?.[kind]?.height ?? TERRAIN_HEIGHTS[kind] ?? 0.08)
}

function resourceColor(snapshot: Snapshot, kind: number) {
  return String(scenarioCatalog(snapshot, 'resources')?.[kind]?.color ?? RESOURCE_COLORS[kind] ?? '#d8a75b')
}

interface SceneProps {
  snapshot: Snapshot
  events: WorldEvent[]
  field: FieldMode
  selectedAgent: string | null
  selectedArtifact: string | null
  selectedTile: [number, number] | null
  showResources: boolean
  showSignals: boolean
  showTrails: boolean
  trail: Array<[number, number, number]>
  onSelectAgent: (id: string) => void
  onSelectArtifact: (id: string) => void
  onSelectTile: (x: number, y: number) => void
  onClearSelection: () => void
}

function seededNoise(x: number, z: number, seed: number) {
  return (
    Math.sin(x * 0.31 + seed * 0.13) * Math.cos(z * 0.27 - seed * 0.09) * 0.18
    + Math.sin((x + z) * 0.73 + seed) * 0.055
    + Math.cos(x * 1.41 - z * 1.17 + seed * 0.3) * 0.026
  )
}

function tileIndex(snapshot: Snapshot, x: number, z: number) {
  const cx = Math.max(0, Math.min(snapshot.world.width - 1, Math.round(x)))
  const cz = Math.max(0, Math.min(snapshot.world.height - 1, Math.round(z)))
  return cz * snapshot.world.width + cx
}

function tileSurface(snapshot: Snapshot, x: number, z: number) {
  const index = tileIndex(snapshot, x, z)
  const kind = snapshot.world.terrain[index] ?? 2
  const base = terrainHeight(snapshot, kind)
  const roughness = kind <= 1 ? 0.24 : kind >= 7 ? 0.1 : 1
  return base + seededNoise(x, z, snapshot.seed) * roughness
}

function vertexSurface(snapshot: Snapshot, x: number, z: number) {
  const samples = [
    tileSurface(snapshot, x - 0.5, z - 0.5),
    tileSurface(snapshot, x + 0.5, z - 0.5),
    tileSurface(snapshot, x - 0.5, z + 0.5),
    tileSurface(snapshot, x + 0.5, z + 0.5),
  ]
  return samples.reduce((sum, value) => sum + value, 0) / samples.length
}

function fieldColor(snapshot: Snapshot, index: number, mode: FieldMode) {
  if (mode === 'terrain') {
    const color = new THREE.Color(terrainColor(snapshot, snapshot.world.terrain[index]))
    const variation = seededNoise(index % snapshot.world.width, Math.floor(index / snapshot.world.width), snapshot.seed)
    color.offsetHSL(variation * 0.03, variation * 0.1, variation * 0.16)
    return color
  }
  const canonical = snapshot.world[mode as keyof typeof snapshot.world]
  const values = snapshot.world.fields?.[mode] ?? (Array.isArray(canonical) ? canonical : [])
  const value = Number(values[index] ?? 0)
  if (mode === 'moisture') return new THREE.Color().setHSL(0.56, 0.67, 0.25 + value * 0.45)
  if (mode === 'nutrients') return new THREE.Color().setHSL(0.25, 0.61 + value * 0.25, 0.23 + value * 0.48)
  if (mode === 'temperature') return new THREE.Color().setHSL(0.66 - value * 0.64, 0.78, 0.34 + value * 0.24)
  if (mode === 'contamination' || mode === 'toxic_gas') return new THREE.Color().setRGB(0.18 + value * 0.82, 0.27 + value * 0.14, 0.31 + value * 0.31)
  return new THREE.Color().setHSL(0.68 - value * 0.67, 0.74, 0.27 + value * 0.34)
}

function Terrain({ snapshot, field }: Pick<SceneProps, 'snapshot' | 'field'>) {
  const detailTexture = useMemo(() => {
    const size = 96
    const data = new Uint8Array(size * size * 4)
    for (let index = 0; index < size * size; index += 1) {
      const x = index % size
      const z = Math.floor(index / size)
      const value = Math.round(128 + seededNoise(x * 0.4, z * 0.4, snapshot.seed) * 390)
      data[index * 4] = value
      data[index * 4 + 1] = value
      data[index * 4 + 2] = value
      data[index * 4 + 3] = 255
    }
    const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat)
    texture.wrapS = THREE.RepeatWrapping
    texture.wrapT = THREE.RepeatWrapping
    texture.repeat.set(9, 7)
    texture.colorSpace = THREE.NoColorSpace
    texture.needsUpdate = true
    return texture
  }, [snapshot.seed])
  const geometry = useMemo(() => {
    const { width, height } = snapshot.world
    const positions: number[] = []
    const colors: number[] = []
    const uvs: number[] = []
    const indices: number[] = []

    for (let z = 0; z <= height; z += 1) {
      for (let x = 0; x <= width; x += 1) {
        const px = x - width / 2 - 0.5
        const pz = z - height / 2 - 0.5
        positions.push(px, vertexSurface(snapshot, x - 0.5, z - 0.5), pz)
        uvs.push(x / width, z / height)

        const neighbors = [
          tileIndex(snapshot, x - 1, z - 1), tileIndex(snapshot, x, z - 1),
          tileIndex(snapshot, x - 1, z), tileIndex(snapshot, x, z),
        ]
        const color = neighbors.reduce((mixed, index) => mixed.add(fieldColor(snapshot, index, field)), new THREE.Color()).multiplyScalar(0.25)
        colors.push(color.r, color.g, color.b)
      }
    }

    for (let z = 0; z < height; z += 1) {
      for (let x = 0; x < width; x += 1) {
        const a = z * (width + 1) + x
        const b = a + 1
        const c = a + width + 1
        const d = c + 1
        indices.push(a, c, b, b, c, d)
      }
    }

    const surface = new THREE.BufferGeometry()
    surface.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    surface.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
    surface.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
    surface.setIndex(indices)
    surface.computeVertexNormals()
    surface.computeBoundingSphere()
    return surface
  }, [field, snapshot])

  useEffect(() => () => geometry.dispose(), [geometry])
  useEffect(() => () => detailTexture.dispose(), [detailTexture])

  return (
    <mesh geometry={geometry} receiveShadow castShadow>
      <meshStandardMaterial vertexColors roughness={0.9} metalness={0.01} bumpMap={detailTexture} bumpScale={0.055} envMapIntensity={0.18} emissive="#102419" emissiveIntensity={0.12} toneMapped={false} />
    </mesh>
  )
}

function WaterSurface({ snapshot }: { snapshot: Snapshot }) {
  const material = useRef<THREE.ShaderMaterial>(null)
  const geometry = useMemo(() => {
    const positions: number[] = []
    const uvs: number[] = []
    for (let z = 0; z < snapshot.world.height; z += 1) {
      for (let x = 0; x < snapshot.world.width; x += 1) {
        const terrain = snapshot.world.terrain[z * snapshot.world.width + x]
        if (terrain > 1) continue
        const x0 = x - snapshot.world.width / 2 - 0.5
        const x1 = x0 + 1
        const z0 = z - snapshot.world.height / 2 - 0.5
        const z1 = z0 + 1
        const level = terrain === 0 ? -0.08 : -0.045
        positions.push(
          x0, level, z0, x0, level, z1, x1, level, z0,
          x1, level, z0, x0, level, z1, x1, level, z1,
        )
        uvs.push(0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1)
      }
    }
    const water = new THREE.BufferGeometry()
    water.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    water.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2))
    water.computeVertexNormals()
    water.computeBoundingSphere()
    return water
  }, [snapshot])
  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uDeep: { value: new THREE.Color(terrainColor(snapshot, 0)) },
    uShallow: { value: new THREE.Color(terrainColor(snapshot, 1)) },
  }), [snapshot])

  useFrame((state) => {
    if (material.current) material.current.uniforms.uTime.value = state.clock.elapsedTime
  })
  useEffect(() => () => geometry.dispose(), [geometry])

  return (
    <mesh geometry={geometry} renderOrder={2}>
      <shaderMaterial
        ref={material}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        side={THREE.DoubleSide}
        vertexShader={`
          uniform float uTime;
          varying float vWave;
          varying vec2 vUvWater;
          void main() {
            vec3 p = position;
            float broad = sin(p.x * 0.34 + uTime * 0.7) * 0.025;
            float fine = cos(p.z * 0.51 - uTime * 0.9 + p.x * 0.15) * 0.014;
            p.y += broad + fine;
            vWave = broad + fine;
            vUvWater = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
          }
        `}
        fragmentShader={`
          uniform vec3 uDeep;
          uniform vec3 uShallow;
          varying float vWave;
          varying vec2 vUvWater;
          void main() {
            float ripple = 0.5 + 0.5 * sin((vUvWater.x + vUvWater.y) * 85.0 + vWave * 120.0);
            vec3 color = mix(uDeep, uShallow, 0.28 + ripple * 0.13 + vWave * 3.0);
            gl_FragColor = vec4(color, 0.57);
          }
        `}
      />
    </mesh>
  )
}

type ResourceShape = 'cone' | 'sphere' | 'octahedron' | 'icosahedron' | 'cylinder'

interface ResourceLayerProps {
  snapshot: Snapshot
  kind: number
  shape: ResourceShape
  scale: [number, number, number]
  offsetY: number
  color: string
  density?: number
}

function ResourceLayer({ snapshot, kind, shape, scale, offsetY, color, density = 0.54 }: ResourceLayerProps) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const matrix = useMemo(() => new THREE.Matrix4(), [])
  const tint = useMemo(() => new THREE.Color(), [])
  const entries = useMemo(() => snapshot.world.resource_kind.flatMap((resource, index) => {
    if (resource !== kind || snapshot.world.resource_mass[index] <= 0.06) return []
    const chance = Math.abs(Math.sin(index * 91.173 + snapshot.seed * 7.13)) % 1
    return chance < density ? [index] : []
  }), [density, kind, snapshot.seed, snapshot.world.resource_kind, snapshot.world.resource_mass])

  useEffect(() => {
    if (!mesh.current) return
    entries.forEach((worldIndex, instanceIndex) => {
      const x = worldIndex % snapshot.world.width
      const z = Math.floor(worldIndex / snapshot.world.width)
      const hash = Math.abs(Math.sin(worldIndex * 41.77 + snapshot.seed))
      const jitterX = (hash % 1 - 0.5) * 0.54
      const jitterZ = ((hash * 7.31) % 1 - 0.5) * 0.54
      const mass = snapshot.world.resource_mass[worldIndex] ?? 0
      const size = 0.72 + Math.min(3.2, mass) * 0.12
      const px = x + jitterX
      const pz = z + jitterZ
      matrix.compose(
        new THREE.Vector3(px - snapshot.world.width / 2, tileSurface(snapshot, px, pz) + offsetY * size, pz - snapshot.world.height / 2),
        new THREE.Quaternion().setFromEuler(new THREE.Euler((hash % 1 - 0.5) * 0.14, hash * Math.PI * 2, ((hash * 3.1) % 1 - 0.5) * 0.14)),
        new THREE.Vector3(scale[0] * size, scale[1] * size, scale[2] * size),
      )
      mesh.current!.setMatrixAt(instanceIndex, matrix)
      tint.set(color).offsetHSL((hash % 1 - 0.5) * 0.025, 0, (hash % 1 - 0.5) * 0.08)
      mesh.current!.setColorAt(instanceIndex, tint)
    })
    mesh.current.instanceMatrix.needsUpdate = true
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true
    mesh.current.computeBoundingSphere()
  }, [color, entries, matrix, offsetY, scale, snapshot, tint])

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, entries.length]} castShadow={kind > 1} receiveShadow>
      {shape === 'cone' && <coneGeometry args={[1, 2, 7]} />}
      {shape === 'sphere' && <sphereGeometry args={[1, 12, 8]} />}
      {shape === 'octahedron' && <octahedronGeometry args={[1, 0]} />}
      {shape === 'icosahedron' && <icosahedronGeometry args={[1, 0]} />}
      {shape === 'cylinder' && <cylinderGeometry args={[1, 1.08, 2, 8]} />}
      <meshStandardMaterial vertexColors roughness={kind === 6 ? 0.38 : 0.78} metalness={kind === 6 ? 0.26 : 0.01} emissive={color} emissiveIntensity={kind === 6 ? 0.2 : 0.16} />
    </instancedMesh>
  )
}

function BiologicalResources({ snapshot }: { snapshot: Snapshot }) {
  if (snapshot.scenario ?? snapshot.world.scenario) {
    return (
      <group>
        {Array.from({ length: 8 }, (_, index) => index + 1).map((kind) => (
          <ResourceLayer key={kind} snapshot={snapshot} kind={kind} shape={kind % 2 ? 'icosahedron' : 'octahedron'} scale={[0.18, 0.16 + (kind % 3) * 0.04, 0.18]} offsetY={0.18} color={resourceColor(snapshot, kind)} density={0.62} />
        ))}
      </group>
    )
  }
  return (
    <group>
      <ResourceLayer snapshot={snapshot} kind={1} shape="cone" scale={[0.09, 0.38, 0.09]} offsetY={0.38} color={RESOURCE_COLORS[1]} density={0.68} />
      <ResourceLayer snapshot={snapshot} kind={2} shape="sphere" scale={[0.2, 0.08, 0.17]} offsetY={0.08} color={RESOURCE_COLORS[2]} />
      <ResourceLayer snapshot={snapshot} kind={3} shape="cylinder" scale={[0.07, 0.2, 0.07]} offsetY={0.2} color="#d8c8a9" density={0.68} />
      <ResourceLayer snapshot={snapshot} kind={3} shape="sphere" scale={[0.23, 0.09, 0.23]} offsetY={0.43} color={RESOURCE_COLORS[3]} density={0.68} />
      <ResourceLayer snapshot={snapshot} kind={4} shape="icosahedron" scale={[0.19, 0.12, 0.16]} offsetY={0.13} color={RESOURCE_COLORS[4]} />
      <ResourceLayer snapshot={snapshot} kind={5} shape="cone" scale={[0.09, 0.46, 0.09]} offsetY={0.44} color={RESOURCE_COLORS[5]} density={0.72} />
      <ResourceLayer snapshot={snapshot} kind={6} shape="octahedron" scale={[0.15, 0.38, 0.15]} offsetY={0.36} color={RESOURCE_COLORS[6]} density={0.62} />
    </group>
  )
}

function Stations({ snapshot }: { snapshot: Snapshot }) {
  const points = useMemo(() => snapshot.world.stations.flatMap((kind, index) => kind > 0 ? [{ kind, index }] : []), [snapshot.world.stations])
  return (
    <group>
      {points.map(({ kind, index }) => {
        const gx = index % snapshot.world.width
        const gz = Math.floor(index / snapshot.world.width)
        const x = gx - snapshot.world.width / 2
        const z = gz - snapshot.world.height / 2
        const y = tileSurface(snapshot, gx, gz)
        return (
          <group key={index} position={[x, y + 0.4, z]}>
            <mesh position={[0, -0.31, 0]} receiveShadow>
              <cylinderGeometry args={[0.57, 0.62, 0.12, 12]} />
              <meshStandardMaterial color="#465d56" roughness={0.72} metalness={0.3} />
            </mesh>
            <mesh castShadow>
              <cylinderGeometry args={[0.34, 0.42, 0.62, 10]} />
              <meshStandardMaterial color="#a8bab0" roughness={0.42} metalness={0.46} />
            </mesh>
            <mesh position={[0, 0.11, 0.34]}>
              <ringGeometry args={[0.09, 0.14, 20]} />
              <meshBasicMaterial color={AGENT_COLORS[kind % AGENT_COLORS.length]} toneMapped={false} />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}

function Agents({ snapshot, selectedAgent, onSelectAgent }: Pick<SceneProps, 'snapshot' | 'selectedAgent' | 'onSelectAgent'>) {
  const bodies = useRef<THREE.InstancedMesh>(null)
  const visors = useRef<THREE.InstancedMesh>(null)
  const matrix = useMemo(() => new THREE.Matrix4(), [])
  const color = useMemo(() => new THREE.Color(), [])
  const current = useRef(snapshot.agents.x.map((x, index) => new THREE.Vector3(x, 0, snapshot.agents.y[index])))
  const targets = useRef(snapshot.agents.x.map((x, index) => new THREE.Vector3(x, 0, snapshot.agents.y[index])))

  useEffect(() => {
    targets.current = snapshot.agents.x.map((x, index) => new THREE.Vector3(x, 0, snapshot.agents.y[index]))
    while (current.current.length < targets.current.length) current.current.push(targets.current[current.current.length].clone())
    if (visors.current) {
      snapshot.agents.visor.forEach((visor, index) => {
        color.set(AGENT_COLORS[visor % AGENT_COLORS.length])
        visors.current!.setColorAt(index, color)
      })
      if (visors.current.instanceColor) visors.current.instanceColor.needsUpdate = true
    }
  }, [color, snapshot.agents.visor, snapshot.agents.x, snapshot.agents.y])

  useFrame((state, delta) => {
    if (!bodies.current || !visors.current) return
    const amount = 1 - Math.exp(-delta * 7)
    for (let index = 0; index < snapshot.agents.display_count; index += 1) {
      const point = current.current[index]
      point.lerp(targets.current[index], amount)
      const x = point.x - snapshot.world.width / 2
      const z = point.z - snapshot.world.height / 2
      const ground = tileSurface(snapshot, point.x, point.z)
      const bob = Math.sin(state.clock.elapsedTime * 3.2 + index) * 0.025
      matrix.compose(new THREE.Vector3(x, ground + 0.59 + bob, z), new THREE.Quaternion(), new THREE.Vector3(0.57, 0.65, 0.57))
      bodies.current.setMatrixAt(index, matrix)
      matrix.compose(new THREE.Vector3(x, ground + 0.74 + bob, z + 0.4), new THREE.Quaternion(), new THREE.Vector3(0.21, 0.21, 0.21))
      visors.current.setMatrixAt(index, matrix)
    }
    bodies.current.instanceMatrix.needsUpdate = true
    visors.current.instanceMatrix.needsUpdate = true
  })

  const selectedIndex = selectedAgent ? snapshot.agents.ids.indexOf(selectedAgent) : -1
  const selectedX = selectedIndex >= 0 ? snapshot.agents.x[selectedIndex] : 0
  const selectedZ = selectedIndex >= 0 ? snapshot.agents.y[selectedIndex] : 0
  const selectedPosition = selectedIndex >= 0 ? [
    selectedX - snapshot.world.width / 2,
    tileSurface(snapshot, selectedX, selectedZ) + 0.035,
    selectedZ - snapshot.world.height / 2,
  ] as [number, number, number] : null

  const select = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    if (event.instanceId !== undefined) onSelectAgent(snapshot.agents.ids[event.instanceId])
  }

  return (
    <group>
      <instancedMesh ref={bodies} args={[undefined, undefined, snapshot.agents.display_count]} castShadow onClick={select}>
        <capsuleGeometry args={[0.52, 0.72, 5, 10]} />
        <meshStandardMaterial color="#e3dfcc" roughness={0.48} metalness={0.2} />
      </instancedMesh>
      <instancedMesh ref={visors} args={[undefined, undefined, snapshot.agents.display_count]} onClick={select}>
        <circleGeometry args={[1, 18]} />
        <meshBasicMaterial vertexColors toneMapped={false} />
      </instancedMesh>
      {selectedPosition && (
        <mesh position={selectedPosition} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.62, 0.79, 48]} />
          <meshBasicMaterial color="#fff0a0" transparent opacity={0.98} side={THREE.DoubleSide} toneMapped={false} />
        </mesh>
      )}
    </group>
  )
}

function actionColor(action: number) {
  if ([11, 17, 18].includes(action)) return '#67e6c0'
  if ([8, 9, 10].includes(action)) return '#f0c966'
  if ([14, 15, 19].includes(action)) return '#c894f0'
  if ([2, 6, 7, 12, 13].includes(action)) return '#77b9ec'
  if ([3, 4, 5, 20].includes(action)) return '#8fd67b'
  return '#d9e4df'
}

function ActionIndicators({ snapshot }: { snapshot: Snapshot }) {
  const mesh = useRef<THREE.InstancedMesh>(null)
  const matrix = useMemo(() => new THREE.Matrix4(), [])
  const quaternion = useMemo(
    () => new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)),
    [],
  )
  const color = useMemo(() => new THREE.Color(), [])
  const entries = useMemo(() => snapshot.agents.last_action.flatMap((action, index) => (
    action > 0 && index < snapshot.agents.display_count && snapshot.agents.active?.[index] !== false
      ? [{ action, index }]
      : []
  )), [snapshot.agents.active, snapshot.agents.display_count, snapshot.agents.last_action])

  useEffect(() => {
    if (!mesh.current) return
    entries.forEach(({ action, index }, instance) => {
      const gx = snapshot.agents.x[index]
      const gz = snapshot.agents.y[index]
      const size = action === 1 ? 0.54 : 0.68
      matrix.compose(
        new THREE.Vector3(
          gx - snapshot.world.width / 2,
          tileSurface(snapshot, gx, gz) + 0.055,
          gz - snapshot.world.height / 2,
        ),
        quaternion,
        new THREE.Vector3(size, size, size),
      )
      mesh.current!.setMatrixAt(instance, matrix)
      color.set(actionColor(action))
      mesh.current!.setColorAt(instance, color)
    })
    mesh.current.instanceMatrix.needsUpdate = true
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true
  }, [color, entries, matrix, quaternion, snapshot])

  if (!entries.length) return null
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, entries.length]} renderOrder={5}>
      <ringGeometry args={[0.68, 0.82, 24]} />
      <meshBasicMaterial vertexColors transparent opacity={0.78} depthWrite={false} toneMapped={false} side={THREE.DoubleSide} />
    </instancedMesh>
  )
}

export function artifactDisplayOffset(snapshot: Snapshot, artifactIndex: number): [number, number] {
  const x = snapshot.artifacts.x[artifactIndex]
  const y = snapshot.artifacts.y[artifactIndex]
  const colocated = snapshot.artifacts.ids.flatMap((_, index) => (
    snapshot.artifacts.x[index] === x && snapshot.artifacts.y[index] === y ? [index] : []
  ))
  if (colocated.length < 2) return [0, 0]
  const slot = colocated.indexOf(artifactIndex)
  const angle = -Math.PI / 2 + slot * Math.PI * 2 / colocated.length
  const radius = 0.68 + Math.floor(slot / 7) * 0.16
  return [Math.cos(angle) * radius, Math.sin(angle) * radius]
}

function Artifacts({ snapshot, selectedArtifact, onSelectArtifact }: Pick<SceneProps, 'snapshot' | 'selectedArtifact' | 'onSelectArtifact'>) {
  return (
    <group>
      {snapshot.artifacts.ids.map((id, index) => {
        const selected = selectedArtifact === id
        const maturity = snapshot.artifacts.maturity[index]
        const gx = snapshot.artifacts.x[index]
        const gz = snapshot.artifacts.y[index]
        const [displayOffsetX, displayOffsetZ] = artifactDisplayOffset(snapshot, index)
        const geometry = snapshot.artifacts.geometry[index] ?? {}
        const layers = Math.max(1, Math.round(geometry.layers ?? 1))
        const area = geometry.surface_area ?? 1
        const channels = geometry.channel_density ?? 0.25
        const anisotropy = geometry.anisotropy ?? 0.25
        const branching = geometry.branching ?? 0.25
        const connectivity = geometry.connectivity ?? 0.5
        const curvature = geometry.curvature ?? 0.25
        const modularity = geometry.modularity ?? 0.25
        const branchCount = 2 + Math.round(branching * 8)
        const visibleLayers = Math.min(5, layers)
        const artifactColor = new THREE.Color().setHSL(
          0.36 + curvature * 0.18,
          0.48 + channels * 0.28,
          0.48 + modularity * 0.15,
        ).getStyle()
        return (
          <group
            key={id}
            position={[gx + displayOffsetX - snapshot.world.width / 2, tileSurface(snapshot, gx, gz) + 0.37, gz + displayOffsetZ - snapshot.world.height / 2]}
            scale={(0.58 + maturity * 0.42) * Math.sqrt(Math.max(0.4, area))}
            onClick={(event) => { event.stopPropagation(); onSelectArtifact(id) }}
          >
            <mesh scale={[0.72 + anisotropy * 0.7, 0.72 + connectivity * 0.45, 0.72 + (1 - anisotropy) * 0.7]} castShadow>
              <icosahedronGeometry args={[0.31 + connectivity * 0.16, Math.round(modularity * 2)]} />
              <meshStandardMaterial color={artifactColor} roughness={0.72 - connectivity * 0.35} metalness={channels * 0.18} />
            </mesh>
            {Array.from({ length: branchCount }, (_, branch) => {
              const angle = branch * Math.PI * 2 / branchCount
              const length = 0.34 + branching * 0.42
              return (
                <group key={`branch-${branch}`} rotation={[0, -angle, 0]}>
                  <mesh position={[length * 0.5, curvature * 0.13 * Math.sin(angle * 2), 0]} rotation={[0, 0, Math.PI / 2 + (curvature - 0.5) * 0.28]} castShadow>
                    <cylinderGeometry args={[0.025 + channels * 0.045, 0.045 + connectivity * 0.04, length, 7]} />
                    <meshStandardMaterial color={artifactColor} roughness={0.5} />
                  </mesh>
                  <mesh position={[length, curvature * 0.2 * Math.sin(angle * 2), 0]} castShadow>
                    <sphereGeometry args={[0.055 + modularity * 0.075, 9, 7]} />
                    <meshStandardMaterial color={artifactColor} emissive={artifactColor} emissiveIntensity={0.08 + snapshot.artifacts.performance[index] * 0.25} />
                  </mesh>
                </group>
              )
            })}
            {Array.from({ length: visibleLayers }, (_, layer) => (
              <mesh key={`layer-${layer}`} rotation={[Math.PI / 2, curvature * layer * 0.18, 0]} position={[0, -0.15 + layer * 0.075, 0]}>
                <torusGeometry args={[0.25 + layer * 0.075, 0.012 + channels * 0.014, 5, 24]} />
                <meshStandardMaterial color={artifactColor} transparent opacity={0.5 + connectivity * 0.35} />
              </mesh>
            ))}
            {selected && (
              <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.29, 0]}>
                <ringGeometry args={[0.69, 0.79, 36]} />
                <meshBasicMaterial color="#fff0a0" side={THREE.DoubleSide} toneMapped={false} />
              </mesh>
            )}
          </group>
        )
      })}
    </group>
  )
}

function SignalLines({ snapshot, events }: Pick<SceneProps, 'snapshot' | 'events'>) {
  const lines = events.slice(-240).flatMap((event, eventIndex) => {
    if (event.kind !== 'message_delivered') return []
    const age = snapshot.tick - event.tick
    if (age < 0 || age > 5) return []
    const sender = String(event.payload.sender ?? '')
    const senderIndex = snapshot.agents.ids.indexOf(sender)
    if (senderIndex < 0) return []
    return ((event.payload.recipients as string[] | undefined) ?? []).flatMap((recipient) => {
      const recipientIndex = snapshot.agents.ids.indexOf(recipient)
      if (recipientIndex < 0) return []
      const sx = snapshot.agents.x[senderIndex]
      const sz = snapshot.agents.y[senderIndex]
      const rx = snapshot.agents.x[recipientIndex]
      const rz = snapshot.agents.y[recipientIndex]
      return [{
        key: `${event.tick}-${eventIndex}-${sender}-${recipient}`,
        opacity: Math.max(0.16, 0.88 - age * 0.14),
        points: [
          [sx - snapshot.world.width / 2, tileSurface(snapshot, sx, sz) + 1.08, sz - snapshot.world.height / 2],
          [rx - snapshot.world.width / 2, tileSurface(snapshot, rx, rz) + 1.08, rz - snapshot.world.height / 2],
        ] as Array<[number, number, number]>,
      }]
    })
  })
  return <>{lines.map((line) => <Line key={line.key} points={line.points} color="#67e6c0" lineWidth={1.6} transparent opacity={line.opacity} />)}</>
}

function ArtifactInteractionLines({ snapshot, events }: Pick<SceneProps, 'snapshot' | 'events'>) {
  const styles: Record<string, string> = {
    artifact_built: '#f0c966',
    artifact_program_installed: '#c894f0',
    artifact_repaired: '#8fd67b',
    artifact_dismantled: '#ef7e72',
  }
  const lines = events.slice(-320).flatMap((event, eventIndex) => {
    const color = styles[event.kind]
    if (!color) return []
    const age = snapshot.tick - event.tick
    if (age < 0 || age > 5) return []
    const agentId = String(event.payload.agent ?? event.payload.author ?? '')
    const artifactId = String(event.payload.artifact_id ?? event.payload.artifact ?? '')
    const agentIndex = snapshot.agents.ids.indexOf(agentId)
    const artifactIndex = snapshot.artifacts.ids.indexOf(artifactId)
    if (agentIndex < 0 || artifactIndex < 0) return []
    const ax = snapshot.agents.x[agentIndex]
    const az = snapshot.agents.y[agentIndex]
    const tx = snapshot.artifacts.x[artifactIndex]
    const tz = snapshot.artifacts.y[artifactIndex]
    const start: [number, number, number] = [
      ax - snapshot.world.width / 2,
      tileSurface(snapshot, ax, az) + 1.02,
      az - snapshot.world.height / 2,
    ]
    const end: [number, number, number] = [
      tx - snapshot.world.width / 2,
      tileSurface(snapshot, tx, tz) + 0.72,
      tz - snapshot.world.height / 2,
    ]
    const midpoint: [number, number, number] = [
      (start[0] + end[0]) / 2,
      Math.max(start[1], end[1]) + 0.85,
      (start[2] + end[2]) / 2,
    ]
    return [{
      key: `${event.kind}-${event.tick}-${eventIndex}-${agentId}-${artifactId}`,
      color,
      opacity: Math.max(0.18, 0.9 - age * 0.14),
      points: [start, midpoint, end],
    }]
  })
  return <>{lines.map((line) => (
    <Line key={line.key} points={line.points} color={line.color} lineWidth={2} transparent opacity={line.opacity} />
  ))}</>
}

function Spores({ snapshot }: { snapshot: Snapshot }) {
  const points = useRef<THREE.Points>(null)
  const positions = useMemo(() => {
    const values = new Float32Array(360 * 3)
    for (let i = 0; i < 360; i += 1) {
      values[i * 3] = ((i * 17) % snapshot.world.width) - snapshot.world.width / 2
      values[i * 3 + 1] = 0.9 + ((i * 13) % 34) / 11
      values[i * 3 + 2] = ((i * 29) % snapshot.world.height) - snapshot.world.height / 2
    }
    return values
  }, [snapshot.world.height, snapshot.world.width])
  useFrame((state) => {
    if (points.current) points.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.06) * 0.025
  })
  return (
    <points ref={points}>
      <bufferGeometry><bufferAttribute attach="attributes-position" args={[positions, 3]} /></bufferGeometry>
      <pointsMaterial color="#def3cf" size={0.045} transparent opacity={0.34} depthWrite={false} />
    </points>
  )
}

function Atmosphere() {
  return (
    <mesh scale={115} renderOrder={-10}>
      <sphereGeometry args={[1, 32, 18]} />
      <shaderMaterial
        side={THREE.BackSide}
        depthWrite={false}
        vertexShader={`
          varying vec3 vDirection;
          void main() {
            vDirection = position;
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `}
        fragmentShader={`
          varying vec3 vDirection;
          void main() {
            float horizon = smoothstep(-0.25, 0.72, normalize(vDirection).y);
            vec3 low = vec3(0.42, 0.57, 0.53);
            vec3 high = vec3(0.10, 0.24, 0.25);
            gl_FragColor = vec4(mix(low, high, horizon), 1.0);
          }
        `}
      />
    </mesh>
  )
}

function World({ onClearSelection, ...props }: SceneProps) {
  const solar = props.snapshot.world.solar.reduce((sum, value) => sum + value, 0) / props.snapshot.world.solar.length
  const raisedTrail = props.trail.map(([x, , z]) => {
    const gx = x + props.snapshot.world.width / 2
    const gz = z + props.snapshot.world.height / 2
    return [x, tileSurface(props.snapshot, gx, gz) + 0.17, z] as [number, number, number]
  })
  const selectedTilePosition = props.selectedTile ? [
    props.selectedTile[0] - props.snapshot.world.width / 2,
    tileSurface(props.snapshot, props.selectedTile[0], props.selectedTile[1]) + 0.08,
    props.selectedTile[1] - props.snapshot.world.height / 2,
  ] as [number, number, number] : null

  const selectTile = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    const x = Math.max(0, Math.min(props.snapshot.world.width - 1, Math.round(event.point.x + props.snapshot.world.width / 2)))
    const y = Math.max(0, Math.min(props.snapshot.world.height - 1, Math.round(event.point.z + props.snapshot.world.height / 2)))
    props.onSelectTile(x, y)
  }
  return (
    <>
      <color attach="background" args={['#294746']} />
      <fog attach="fog" args={['#6f8981', 92, 165]} />
      <Atmosphere />
      <hemisphereLight args={['#e5f5ea', '#26382e', 0.82 + solar * 0.32]} />
      <directionalLight
        position={[25, 38, 18]}
        intensity={1.45 + solar * 0.5}
        color="#fff0c9"
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-34}
        shadow-camera-right={34}
        shadow-camera-top={28}
        shadow-camera-bottom={-28}
        shadow-camera-near={1}
        shadow-camera-far={95}
        shadow-bias={-0.00035}
      />
      <group onClick={selectTile} onPointerMissed={onClearSelection}>
        <Terrain snapshot={props.snapshot} field={props.field} />
        <WaterSurface snapshot={props.snapshot} />
        {props.showResources && <BiologicalResources snapshot={props.snapshot} />}
        <Stations snapshot={props.snapshot} />
        <Artifacts snapshot={props.snapshot} selectedArtifact={props.selectedArtifact} onSelectArtifact={props.onSelectArtifact} />
        <Agents snapshot={props.snapshot} selectedAgent={props.selectedAgent} onSelectAgent={props.onSelectAgent} />
        <ActionIndicators snapshot={props.snapshot} />
        {selectedTilePosition && (
          <mesh position={selectedTilePosition} rotation={[-Math.PI / 2, 0, 0]}>
            <ringGeometry args={[0.43, 0.53, 36]} />
            <meshBasicMaterial color="#fff0a0" transparent opacity={0.94} side={THREE.DoubleSide} toneMapped={false} />
          </mesh>
        )}
        {props.showSignals && <SignalLines snapshot={props.snapshot} events={props.events} />}
        {props.showSignals && <ArtifactInteractionLines snapshot={props.snapshot} events={props.events} />}
        {props.showTrails && raisedTrail.length > 1 && <Line points={raisedTrail} color="#ffec84" lineWidth={2.2} transparent opacity={0.9} />}
        <Spores snapshot={props.snapshot} />
      </group>
      <MapControls makeDefault enableDamping dampingFactor={0.08} minZoom={8} maxZoom={55} maxPolarAngle={Math.PI / 2.35} minPolarAngle={0.5} target={[2, 0, 0]} />
    </>
  )
}

export function WorldScene(props: SceneProps) {
  return (
    <Canvas
      shadows="basic"
      orthographic
      camera={{ position: [36, 30, 39], zoom: 15, near: 0.1, far: 220 }}
      dpr={[1, 1.65]}
      gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      onCreated={({ gl }) => {
        gl.toneMapping = THREE.ACESFilmicToneMapping
        gl.toneMappingExposure = 0.98
        gl.outputColorSpace = THREE.SRGBColorSpace
      }}
    >
      <World {...props} />
    </Canvas>
  )
}
