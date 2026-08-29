import { FIELD_OVERLAYS } from './constants.js'
import { toast } from './hud.js'

const KEY_DIRECTIONS = {
  ArrowUp: 'NORTH',
  ArrowDown: 'SOUTH',
  ArrowLeft: 'WEST',
  ArrowRight: 'EAST',
  Up: 'NORTH',
  Down: 'SOUTH',
  Left: 'WEST',
  Right: 'EAST',
  w: 'NORTH',
  s: 'SOUTH',
  a: 'WEST',
  d: 'EAST',
  W: 'NORTH',
  S: 'SOUTH',
  A: 'WEST',
  D: 'EAST',
}

export function bindInput({ canvas, state, renderer, send, act }) {
  const chatInput = document.getElementById('chat-input')

  const largestHeldResource = () => {
    const inventory = state.humanObs?.self?.inventory
    if (!inventory) return null
    let best = null
    let bestMass = 0
    for (const [name, mass] of Object.entries(inventory)) {
      if (Number(mass) > bestMass) {
        best = name
        bestMass = Number(mass)
      }
    }
    return best
  }

  const verbAction = (verb) => {
    if (verb === 'DEPOSIT') {
      const resource = largestHeldResource()
      if (!resource) {
        toast('nothing to deposit — harvest something first', true)
        return
      }
      act({ verb: 'DEPOSIT', resource: resource.toUpperCase() })
      return
    }
    act({ verb })
  }

  window.addEventListener('keydown', (event) => {
    if (document.activeElement === chatInput) {
      if (event.key === 'Escape') chatInput.blur()
      return
    }
    const direction = KEY_DIRECTIONS[event.key]
    if (direction) {
      event.preventDefault()
      act({ verb: 'MOVE', direction })
      return
    }
    switch (event.key) {
      case ' ':
        event.preventDefault()
        act({ verb: 'WAIT' })
        break
      case 'e':
      case 'E':
        act({ verb: 'INSPECT' })
        break
      case 'h':
      case 'H':
        act({ verb: 'HARVEST' })
        break
      case 'g':
      case 'G':
        verbAction('DEPOSIT')
        break
      case 't':
      case 'T':
        act({ verb: 'TEST' })
        break
      case 'p':
      case 'P':
        send({ command: 'toggle_pause' })
        break
      case 'n':
      case 'N':
        send({ command: 'step' })
        break
      case '-':
      case '_':
        send({
          command: 'set_speed',
          speed_multiplier: Math.max(0.25, (state.control?.speed_multiplier ?? 1) / 2),
        })
        break
      case '+':
      case '=':
        send({
          command: 'set_speed',
          speed_multiplier: Math.min(8, (state.control?.speed_multiplier ?? 1) * 2),
        })
        break
      case 'f':
      case 'F':
        state.fieldOverlay = (state.fieldOverlay + 1) % FIELD_OVERLAYS.length
        break
      case 'm':
      case 'M':
        state.follow = !state.follow
        toast(state.follow ? 'camera: following you' : 'camera: free', false)
        break
      case 'c':
      case 'C':
      case 'Enter':
        event.preventDefault()
        chatInput.focus()
        break
      case 'Escape':
        state.selected = null
        break
      default:
        break
    }
  })

  // Mouse: click to select, drag to pan, wheel to zoom.
  let dragging = null
  canvas.addEventListener('mousedown', (event) => {
    dragging = { x: event.clientX, y: event.clientY, moved: false }
  })
  window.addEventListener('mousemove', (event) => {
    if (!dragging) return
    const dx = event.clientX - dragging.x
    const dy = event.clientY - dragging.y
    if (Math.abs(dx) + Math.abs(dy) > 3) {
      dragging.moved = true
      state.follow = false
      renderer.camera.x -= dx / renderer.camera.zoom
      renderer.camera.y -= dy / renderer.camera.zoom
      dragging.x = event.clientX
      dragging.y = event.clientY
    }
  })
  window.addEventListener('mouseup', (event) => {
    if (!dragging) return
    if (!dragging.moved) {
      const rect = canvas.getBoundingClientRect()
      const tile = renderer.screenToTile(
        event.clientX - rect.left,
        event.clientY - rect.top,
        state.snapshot?.world,
      )
      state.selected = tile
    }
    dragging = null
  })
  canvas.addEventListener('wheel', (event) => {
    event.preventDefault()
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15
    renderer.camera.zoom = Math.max(6, Math.min(46, renderer.camera.zoom * factor))
  }, { passive: false })

  // Panel buttons
  for (const button of document.querySelectorAll('#dpad button[data-dir]')) {
    button.addEventListener('click', () => act({ verb: 'MOVE', direction: button.dataset.dir }))
  }
  for (const button of document.querySelectorAll('button[data-act]')) {
    button.addEventListener('click', () => verbAction(button.dataset.act))
  }

  document.getElementById('join-btn').addEventListener('click', () => {
    send({ command: state.possessed ? 'release' : 'join' })
  })

  const chatForm = document.getElementById('chat-form')
  chatForm.addEventListener('submit', (event) => {
    event.preventDefault()
    const text = chatInput.value.trim()
    if (!text) return
    act({ verb: 'COMMUNICATE', message: text })
    chatInput.value = ''
    chatInput.blur()
  })
  // Belt and braces: some environments do not run implicit form submission
  // for synthetic Enter key events.
  chatInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault()
      if (chatForm.requestSubmit) chatForm.requestSubmit()
      else chatForm.dispatchEvent(new Event('submit', { cancelable: true }))
    }
  })
}
