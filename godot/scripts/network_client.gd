extends Node

signal packet_received(payload: Dictionary)
signal connection_changed(connected: bool, detail: String)

@export var server_url := "ws://127.0.0.1:8765/ws"
@export var reconnect_seconds := 2.0

var socket := WebSocketPeer.new()
var reconnect_at := 0.0
var was_open := false


func _ready() -> void:
	var override_url := OS.get_environment("BIOFOUNDRY_SERVER_URL")
	if not override_url.is_empty():
		server_url = override_url
	_connect_socket()


func _process(_delta: float) -> void:
	socket.poll()
	var state := socket.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN:
		if not was_open:
			was_open = true
			connection_changed.emit(true, "LIVE")
		while socket.get_available_packet_count() > 0:
			var text := socket.get_packet().get_string_from_utf8()
			var decoded = JSON.parse_string(text)
			if decoded is Dictionary:
				packet_received.emit(decoded)
	elif state == WebSocketPeer.STATE_CLOSED:
		if was_open:
			was_open = false
			connection_changed.emit(false, "RECONNECTING")
		if Time.get_ticks_msec() / 1000.0 >= reconnect_at:
			_connect_socket()


func _connect_socket() -> void:
	socket = WebSocketPeer.new()
	# Full scientific snapshots are larger than the conservative 64 KiB default.
	socket.inbound_buffer_size = 16 * 1024 * 1024
	socket.outbound_buffer_size = 1024 * 1024
	socket.max_queued_packets = 128
	var error := socket.connect_to_url(server_url)
	reconnect_at = Time.get_ticks_msec() / 1000.0 + reconnect_seconds
	if error != OK:
		connection_changed.emit(false, "SERVER OFFLINE")


func send_command(payload: Dictionary) -> void:
	if socket.get_ready_state() != WebSocketPeer.STATE_OPEN:
		connection_changed.emit(false, "COMMAND FAILED — SERVER OFFLINE")
		return
	socket.send_text(JSON.stringify(payload))
