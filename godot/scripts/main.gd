extends Node2D

@onready var world_view: Node2D = $WorldView
@onready var network_client: Node = $NetworkClient
@onready var status_label: Label = $HUD/TopBar/Status
@onready var metrics_label: Label = $HUD/MetricsPanel/Metrics
@onready var inspector_label: Label = $HUD/InteractionPanel/Inspector
@onready var mode_label: Label = $HUD/ModePanel/Mode
@onready var events_label: Label = $HUD/EventPanel/Events

const ACTION_NAMES := [
	"WAIT", "MOVE", "INSPECT", "HARVEST", "DEPOSIT", "OPERATE", "TEST",
	"PROPOSE RECIPE", "BUILD", "REPAIR", "DISMANTLE", "COMMUNICATE", "PUBLISH",
	"DEPOSIT INSIGHT", "WRITE PROGRAM", "FORK PROGRAM", "CLAIM TASK", "TEACH",
	"TRADE", "COMBINE DESIGN", "METABOLIZE",
]

var current_snapshot: Dictionary = {}
var control_state: Dictionary = {}
var event_history: Array = []
var selected_agent := -1
var zoom_level := 1.0
var capture_path := ""
var capture_started := false
var capture_countdown := -1


func _ready() -> void:
	network_client.packet_received.connect(_on_packet)
	network_client.connection_changed.connect(_on_connection_changed)
	capture_path = OS.get_environment("BIOFOUNDRY_CAPTURE")
	if not capture_path.is_empty():
		print("BIOFOUNDRY_CAPTURE_REQUESTED:", capture_path)


func _process(_delta: float) -> void:
	if capture_countdown > 0:
		capture_countdown -= 1
	elif capture_countdown == 0:
		capture_countdown = -1
		var image := get_viewport().get_texture().get_image()
		var error := image.save_png(capture_path)
		if error == OK:
			print("BIOFOUNDRY_CAPTURED:", capture_path)
		else:
			push_error("Could not save BioFoundry capture: %s" % error_string(error))


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		selected_agent = world_view.pick_agent(event.position)
		world_view.set_selected_agent(selected_agent)
		_update_inspector()
		return
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_1:
				world_view.set_view_mode(0)
				_update_mode_text(0)
			KEY_2:
				world_view.set_view_mode(1)
				_update_mode_text(1)
			KEY_3:
				world_view.set_view_mode(2)
				_update_mode_text(2)
			KEY_EQUAL, KEY_PLUS:
				zoom_level = min(1.45, zoom_level + 0.08)
				world_view.scale = Vector2.ONE * zoom_level
			KEY_MINUS:
				zoom_level = max(0.65, zoom_level - 0.08)
				world_view.scale = Vector2.ONE * zoom_level
			KEY_SPACE:
				network_client.send_command({"command": "toggle_pause"})
			KEY_N:
				network_client.send_command({"command": "step"})
			KEY_COMMA:
				_set_speed(float(control_state.get("speed_multiplier", 1.0)) * 0.5)
			KEY_PERIOD:
				_set_speed(float(control_state.get("speed_multiplier", 1.0)) * 2.0)
			KEY_UP:
				_send_manual_action(1, 1)
			KEY_RIGHT:
				_send_manual_action(1, 2)
			KEY_DOWN:
				_send_manual_action(1, 3)
			KEY_LEFT:
				_send_manual_action(1, 4)
			KEY_H:
				_send_manual_action(3)
			KEY_E:
				_send_manual_action(2)
			KEY_R:
				_send_manual_action(9)
			KEY_X:
				_send_manual_action(10)
			KEY_ESCAPE:
				selected_agent = -1
				world_view.set_selected_agent(-1)
				_update_inspector()


func _on_connection_changed(connected: bool, detail: String) -> void:
	status_label.text = ("● " if connected else "○ ") + detail
	status_label.modulate = Color("71e0bd") if connected else Color("edac59")


func _on_packet(payload: Dictionary) -> void:
	var packet_type := str(payload.get("type", ""))
	if payload.has("control"):
		control_state = payload.get("control", {})
	if packet_type in ["snapshot", "frame", "complete"]:
		current_snapshot = payload.get("snapshot", {})
		world_view.set_snapshot(current_snapshot)
		world_view.set_events(payload.get("events", []))
		_update_metrics()
		_update_events(payload.get("events", []))
		_update_inspector()
		if not capture_path.is_empty() and not capture_started:
			capture_started = true
			capture_countdown = 4
	if packet_type == "complete":
		status_label.text = "● EPISODE COMPLETE"
	elif packet_type == "control":
		_update_inspector()


func _update_metrics() -> void:
	if current_snapshot.is_empty():
		return
	var agents: Dictionary = current_snapshot.get("agents", {})
	var artifacts: Dictionary = current_snapshot.get("artifacts", {})
	var metrics: Dictionary = current_snapshot.get("metrics", {})
	metrics_label.text = "TICK       %06d\nAGENTS     %d\nARTIFACTS  %d\nUTILITY    %.2f" % [
		int(current_snapshot.get("tick", 0)),
		int(agents.get("count", 0)),
		int(artifacts.get("count", 0)),
		float(metrics.get("artifact_score", 0.0)),
	]


func _update_events(events: Array) -> void:
	for item in events:
		event_history.append(item)
	if event_history.size() > 14:
		event_history = event_history.slice(event_history.size() - 14)
	var lines := PackedStringArray(["RECENT ACTIVITY"])
	for item in event_history:
		lines.append("• " + _event_summary(item))
	events_label.text = "\n".join(lines)


func _event_summary(value: Variant) -> String:
	if not value is Dictionary:
		return "event"
	var event: Dictionary = value
	var kind := str(event.get("kind", "event"))
	var payload: Dictionary = event.get("payload", {})
	match kind:
		"agent_deliberated":
			return "%s chose %s" % [
				_short_agent(str(payload.get("agent", "agent"))),
				_action_name(int(payload.get("verb", 0))),
			]
		"model_error":
			return "LLM ERROR %s: %s" % [
				_short_agent(str(payload.get("agent", "agent"))),
				str(payload.get("error", "request failed")).left(64),
			]
		"agents_moved":
			var moved: Array = payload.get("indices", [])
			return "%d agents moved" % moved.size()
		"resource_harvested":
			return "%s harvested resource %d" % [
				_short_agent(str(payload.get("agent", "agent"))),
				int(payload.get("resource", 0)),
			]
		"artifact_built":
			return "%s built artifact %d" % [
				_short_agent(str(payload.get("agent", "agent"))),
				int(payload.get("artifact_type", 0)),
			]
		"artifact_program_installed":
			return "%s installed a program" % _short_agent(str(payload.get("agent", "agent")))
		"insight_deposited":
			return "%s deposited an insight" % _short_agent(str(payload.get("author", "agent")))
		_:
			return kind.replace("_", " ").capitalize()


func _update_inspector() -> void:
	var paused := bool(control_state.get("paused", false))
	var speed := float(control_state.get("speed_multiplier", 1.0))
	var model := str(control_state.get("model", "connecting"))
	var errors := int(control_state.get("model_errors", 0))
	var requests := int(control_state.get("model_requests", 0))
	var header := "WORLD CONTROL — %s  %.2fx\nMODEL  %s\nREQUESTS  %d   ERRORS  %d" % [
		"PAUSED" if paused else "RUNNING", speed, model, requests, errors,
	]
	if selected_agent < 0 or current_snapshot.is_empty():
		inspector_label.text = header + "\n\nClick an agent to inspect and control it.\n\n[SPACE] PAUSE   [N] STEP\n[,] / [.] SPEED\n[ARROWS] MOVE   [H] HARVEST\n[E] INSPECT     [R] REPAIR\n[X] DISMANTLE   [ESC] DESELECT"
		return
	var agents: Dictionary = current_snapshot.get("agents", {})
	var ids: Array = agents.get("ids", [])
	if selected_agent >= ids.size():
		selected_agent = -1
		world_view.set_selected_agent(-1)
		_update_inspector()
		return
	var xs: Array = agents.get("x", [])
	var ys: Array = agents.get("y", [])
	var energy: Array = agents.get("energy", [])
	var inventory: Array = agents.get("inventory_total", [])
	var actions: Array = agents.get("last_action", [])
	inspector_label.text = header + "\n\nSELECTED  %s\nPOSITION  (%d, %d)\nENERGY    %.3f\nINVENTORY %.3f\nACTION    %s\n\n[ARROWS] MOVE   [H] HARVEST\n[E] INSPECT  [R] REPAIR  [X] DISMANTLE" % [
		str(ids[selected_agent]), int(xs[selected_agent]), int(ys[selected_agent]),
		float(energy[selected_agent]), float(inventory[selected_agent]),
		_action_name(int(actions[selected_agent])),
	]


func _send_manual_action(verb: int, direction: int = 0) -> void:
	if selected_agent < 0 or current_snapshot.is_empty():
		return
	var agents: Dictionary = current_snapshot.get("agents", {})
	var ids: Array = agents.get("ids", [])
	if selected_agent >= ids.size():
		return
	var xs: Array = agents.get("x", [])
	var ys: Array = agents.get("y", [])
	network_client.send_command({
		"command": "manual_action",
		"agent": str(ids[selected_agent]),
		"action": {
			"verb": verb,
			"direction": direction,
			"target_x": int(xs[selected_agent]),
			"target_y": int(ys[selected_agent]),
		},
	})


func _set_speed(value: float) -> void:
	network_client.send_command({
		"command": "set_speed",
		"speed_multiplier": clampf(value, 0.25, 8.0),
	})


func _action_name(value: int) -> String:
	return ACTION_NAMES[value] if value >= 0 and value < ACTION_NAMES.size() else "UNKNOWN"


func _short_agent(value: String) -> String:
	return value.replace("agent_", "A")


func _update_mode_text(mode: int) -> void:
	var names := ["LIVING WORLD", "SCIENTIFIC FIELDS", "COLLECTIVE INTELLIGENCE"]
	mode_label.text = "VIEW %d — %s\n\n[1] WORLD   [2] SCIENCE   [3] INTELLIGENCE\n[- / +] ZOOM" % [
		mode + 1,
		names[mode],
	]
