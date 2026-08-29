extends Node2D

const TILE_W := 28.0
const TILE_H := 14.0
const ORIGIN := Vector2(770.0, 80.0)

const TERRAIN_COLORS := [
	Color("173f53"), Color("2f7272"), Color("536e3e"), Color("304f42"),
	Color("6c5d35"), Color("73834b"), Color("517c77"), Color("655c4c"),
	Color("52625d"),
]
const RESOURCE_COLORS := [
	Color.TRANSPARENT, Color("6fba78"), Color("e5d1a5"), Color("a68bc6"),
	Color("d69b58"), Color("d2c985"), Color("a9b9bd"), Color("64b8d2"),
	Color("ecb85f"),
]
const VISOR_COLORS := [
	Color("ff765f"), Color("ffc857"), Color("68d5ff"), Color("ad8cff"),
	Color("71e0bd"), Color("9ee06f"), Color("ff79c9"), Color("d99bff"),
	Color("60cbd0"), Color("e8e56a"), Color("ff9c62"), Color("8aa7ff"),
]

var snapshot: Dictionary = {}
var recent_events: Array = []
var view_mode := 0
var selected_agent := -1


func set_snapshot(value: Dictionary) -> void:
	snapshot = value
	queue_redraw()


func set_events(value: Array) -> void:
	recent_events = value
	queue_redraw()


func set_view_mode(value: int) -> void:
	view_mode = clampi(value, 0, 2)
	queue_redraw()


func set_selected_agent(index: int) -> void:
	selected_agent = index
	queue_redraw()


func pick_agent(viewport_position: Vector2) -> int:
	if snapshot.is_empty():
		return -1
	var local_position := to_local(viewport_position)
	var agents: Dictionary = snapshot.get("agents", {})
	var count := int(agents.get("display_count", 0))
	var xs: Array = agents.get("x", [])
	var ys: Array = agents.get("y", [])
	var best := -1
	var best_distance := 22.0
	for index in range(count):
		var center := iso_position(float(xs[index]), float(ys[index])) + Vector2(0, -10)
		var distance := local_position.distance_to(center)
		if distance < best_distance:
			best = index
			best_distance = distance
	return best


func iso_position(x: float, y: float) -> Vector2:
	return ORIGIN + Vector2((x - y) * TILE_W * 0.5, (x + y) * TILE_H * 0.5)


func _catalog_record(world: Dictionary, kind: String, index: int) -> Dictionary:
	var scenario: Dictionary = world.get("scenario", snapshot.get("scenario", {}))
	var catalogs: Dictionary = scenario.get("catalogs", {})
	var records: Array = catalogs.get(kind, [])
	if index >= 0 and index < records.size() and records[index] is Dictionary:
		return records[index]
	return {}


func _catalog_color(world: Dictionary, kind: String, index: int, fallback: Color) -> Color:
	var record := _catalog_record(world, kind, index)
	return Color(str(record.get("color", fallback.to_html())))


func _draw() -> void:
	if snapshot.is_empty():
		_draw_waiting_world()
		return
	var world: Dictionary = snapshot.get("world", {})
	var width := int(world.get("width", 0))
	var height := int(world.get("height", 0))
	var terrain: Array = world.get("terrain", [])
	var resources: Array = world.get("resource_kind", [])
	var masses: Array = world.get("resource_mass", [])
	var stations: Array = world.get("stations", [])
	var moisture: Array = world.get("moisture", [])
	var contamination: Array = world.get("contamination", [])
	var scenario_value: Variant = world.get("scenario", {})
	var scenario_mode: bool = false
	if scenario_value is Dictionary:
		scenario_mode = not scenario_value.is_empty()

	for diagonal in range(width + height - 1):
		for y in range(height):
			var x := diagonal - y
			if x < 0 or x >= width:
				continue
			var index := y * width + x
			var terrain_id := int(terrain[index])
			var fallback: Color = TERRAIN_COLORS[clampi(terrain_id, 0, TERRAIN_COLORS.size() - 1)]
			var color: Color = _catalog_color(world, "terrains", terrain_id, fallback)
			if view_mode == 1:
				var wet := float(moisture[index])
				var dirty := float(contamination[index])
				color = Color(0.12 + dirty * 0.75, 0.18 + wet * 0.45, 0.28 + wet * 0.55, 1.0)
			_draw_tile(x, y, color, terrain_id == 0)
			if int(resources[index]) > 0 and float(masses[index]) > 0.06 and (x * 7 + y * 11) % 3 == 0:
				var resource_id := int(resources[index])
				var resource_fallback: Color = RESOURCE_COLORS[clampi(resource_id, 0, RESOURCE_COLORS.size() - 1)]
				_draw_resource(x, y, resource_id, float(masses[index]), _catalog_color(world, "resources", resource_id, resource_fallback), scenario_mode)
			if int(stations[index]) > 0:
				_draw_station(x, y, int(stations[index]))

	_draw_artifacts(snapshot.get("artifacts", {}))
	_draw_agents(snapshot.get("agents", {}))
	if view_mode == 2:
		_draw_intelligence_overlay()


func _draw_tile(x: int, y: int, color: Color, water: bool) -> void:
	var center := iso_position(x, y)
	var diamond := PackedVector2Array([
		center + Vector2(0, -TILE_H * 0.5),
		center + Vector2(TILE_W * 0.5, 0),
		center + Vector2(0, TILE_H * 0.5),
		center + Vector2(-TILE_W * 0.5, 0),
	])
	draw_colored_polygon(diamond, color)
	draw_polyline(diamond + PackedVector2Array([diamond[0]]), color.lightened(0.08), 0.7)
	if water and (x + y) % 4 == 0:
		draw_line(center + Vector2(-6, 1), center + Vector2(5, -1), Color(0.4, 0.78, 0.86, 0.35), 1.0)


func _draw_resource(x: int, y: int, resource: int, mass: float, color: Color, scenario_mode: bool) -> void:
	var center := iso_position(x, y) + Vector2(0, -4)
	var radius := clampf(1.6 + mass * 0.8, 2.0, 4.8)
	draw_circle(center + Vector2(1.5, 2.0), radius + 1.0, Color(0, 0, 0, 0.28))
	if scenario_mode:
		var rock := PackedVector2Array([
			center + Vector2(0, -radius), center + Vector2(radius, -radius * 0.15),
			center + Vector2(radius * 0.55, radius), center + Vector2(-radius * 0.7, radius * 0.75),
			center + Vector2(-radius, -radius * 0.25),
		])
		draw_colored_polygon(rock, color)
		draw_polyline(rock + PackedVector2Array([rock[0]]), color.lightened(0.2), 1.0)
		return
	match resource:
		1: # Kelp blades
			draw_line(center + Vector2(-2, 3), center + Vector2(-3, -radius), color, 2.0)
			draw_line(center + Vector2(1, 3), center + Vector2(3, -radius * 0.8), color.lightened(0.15), 2.0)
		2: # Shell
			draw_arc(center, radius, PI, TAU, 10, color, 2.5)
			draw_line(center + Vector2(-radius, 0), center + Vector2(radius, 0), color.darkened(0.18), 1.0)
		3: # Mushroom
			draw_line(center + Vector2(0, 3), center + Vector2(0, -1), color.lightened(0.35), 2.0)
			draw_arc(center + Vector2(0, -1), radius, PI, TAU, 10, color, 3.0)
		4: # Chitin beetle silhouette
			_draw_flat_ellipse(center, Vector2(radius, radius * 0.65), color)
			draw_line(center + Vector2(0, -radius * 0.6), center + Vector2(0, radius * 0.6), color.darkened(0.3), 1.0)
		5: # Cellulose grass
			for blade in range(3):
				draw_line(center + Vector2(blade * 2 - 2, 3), center + Vector2(blade - 1, -radius), color, 1.5)
		6: # Mineral crystal
			var crystal := PackedVector2Array([
				center + Vector2(0, -radius), center + Vector2(radius * 0.7, 0),
				center + Vector2(0, radius), center + Vector2(-radius * 0.7, 0),
			])
			draw_colored_polygon(crystal, color)
		_:
			draw_circle(center, radius, color)
	draw_circle(center - Vector2(1.0, 1.0), maxf(0.8, radius * 0.22), color.lightened(0.28))


func _draw_station(x: int, y: int, station: int) -> void:
	var center := iso_position(x, y) + Vector2(0, -8)
	draw_circle(center + Vector2(2, 4), 7.0, Color(0, 0, 0, 0.35))
	draw_rect(Rect2(center - Vector2(6, 7), Vector2(12, 13)), Color("7b8278"), true)
	draw_rect(Rect2(center - Vector2(4, 5), Vector2(8, 7)), Color("aec1ae"), true)
	draw_circle(center - Vector2(0, 3), 2.2, VISOR_COLORS[station % VISOR_COLORS.size()])


func _draw_agents(agents: Dictionary) -> void:
	var count := int(agents.get("display_count", 0))
	var xs: Array = agents.get("x", [])
	var ys: Array = agents.get("y", [])
	var visors: Array = agents.get("visor", [])
	var actions: Array = agents.get("last_action", [])
	var order := []
	for index in range(count):
		order.append(index)
	order.sort_custom(func(a, b): return int(xs[a]) + int(ys[a]) < int(xs[b]) + int(ys[b]))
	for index in order:
		var center := iso_position(float(xs[index]), float(ys[index])) + Vector2(0, -12)
		var visor: Color = VISOR_COLORS[int(visors[index]) % VISOR_COLORS.size()]
		_draw_flat_ellipse(center + Vector2(2, 10), Vector2(8, 3), Color(0, 0, 0, 0.32))
		if index == selected_agent:
			draw_arc(center + Vector2(0, 3), 13.0, 0, TAU, 32, Color("fff09a"), 2.5)
		var body := PackedVector2Array([
			center + Vector2(-6, 9), center + Vector2(-4, -1),
			center + Vector2(4, -1), center + Vector2(6, 9),
		])
		draw_colored_polygon(body, Color("d7d2bd"))
		draw_circle(center - Vector2(0, 4), 6.3, Color("d9d7ca"))
		draw_circle(center - Vector2(0, 4), 4.2, Color("1c2729"))
		draw_circle(center - Vector2(0, 4), 2.2, visor)
		draw_arc(center - Vector2(0, 4), 7.0, 0, TAU, 20, Color("746f63"), 1.0)
		if index < actions.size() and int(actions[index]) > 0:
			var action := int(actions[index])
			var badge_color: Color = VISOR_COLORS[action % VISOR_COLORS.size()]
			var badge := center + Vector2(9, -11)
			draw_circle(badge, 4.5, Color(0.02, 0.05, 0.06, 0.9))
			draw_circle(badge, 3.2, badge_color)


func _draw_flat_ellipse(center: Vector2, radius: Vector2, color: Color) -> void:
	var points := PackedVector2Array()
	for index in range(20):
		var angle := TAU * index / 20.0
		points.append(center + Vector2(cos(angle) * radius.x, sin(angle) * radius.y))
	draw_colored_polygon(points, color)


func _draw_artifacts(artifacts: Dictionary) -> void:
	var count := int(artifacts.get("display_count", 0))
	var kinds: Array = artifacts.get("kind", [])
	var xs: Array = artifacts.get("x", [])
	var ys: Array = artifacts.get("y", [])
	var health: Array = artifacts.get("health", [])
	var maturity: Array = artifacts.get("maturity", [])
	var openings: Array = artifacts.get("open_fraction", [])
	for index in range(count):
		var center := iso_position(float(xs[index]), float(ys[index])) + Vector2(0, -7)
		var alpha := 0.45 + 0.55 * float(health[index])
		match int(kinds[index]):
			1:
				draw_line(center + Vector2(-9, 4), center + Vector2(9, -5), Color(0.8, 0.92, 0.85, alpha), 3.0)
				for drop in range(3):
					draw_circle(center + Vector2(-5 + drop * 5, 7 + drop % 2), 1.5, Color("63c7e0"))
			2:
				var span := 7.0 + 8.0 * float(maturity[index])
				draw_arc(center + Vector2(0, 5), span, PI, TAU, 18, Color(0.75, 0.78, 0.59, alpha), 3.0)
				draw_arc(center + Vector2(0, 5), span - 2, PI, TAU, 18, Color("ddd6a4"), 1.0)
			3:
				draw_arc(center, 8.0, 0, TAU, 24, Color(0.9, 0.76, 0.48, alpha), 3.0)
				draw_line(center + Vector2(-4, 0), center + Vector2(4, 0), Color("fff0b8"), 1.5)
			4:
				var petal := 5.0 + 6.0 * float(openings[index])
				for angle_index in range(6):
					var angle := TAU * angle_index / 6.0
					var tip := center + Vector2(cos(angle), sin(angle)) * petal
					draw_line(center, tip, Color(0.78, 0.9, 0.67, alpha), 3.0)
				draw_circle(center, 3.0, Color("f2c96d"))


func _draw_intelligence_overlay() -> void:
	var agents: Dictionary = snapshot.get("agents", {})
	var ids: Array = agents.get("ids", [])
	var xs: Array = agents.get("x", [])
	var ys: Array = agents.get("y", [])
	var index_by_id := {}
	for index in range(ids.size()):
		index_by_id[str(ids[index])] = index
	for item in recent_events:
		var event: Dictionary = item
		if str(event.get("kind", "")) != "message_delivered":
			continue
		var payload: Dictionary = event.get("payload", {})
		var sender := str(payload.get("sender", ""))
		if not index_by_id.has(sender):
			continue
		var sender_index: int = index_by_id[sender]
		var start := iso_position(float(xs[sender_index]), float(ys[sender_index])) + Vector2(0, -20)
		for recipient in payload.get("recipients", []):
			if not index_by_id.has(str(recipient)):
				continue
			var recipient_index: int = index_by_id[str(recipient)]
			var finish := iso_position(float(xs[recipient_index]), float(ys[recipient_index])) + Vector2(0, -20)
			draw_line(start, finish, Color(0.45, 0.92, 0.82, 0.62), 1.5)
			draw_circle(finish, 2.0, Color("9ff5d7"))
	for insight in snapshot.get("insights", []):
		var note: Dictionary = insight
		var point := iso_position(float(note.get("x", 0)), float(note.get("y", 0))) + Vector2(0, -24)
		draw_circle(point, 4.0, Color(0.96, 0.72, 0.30, 0.85))
		draw_arc(point, 7.0, 0, TAU, 16, Color(0.96, 0.72, 0.30, 0.4), 1.0)


func _draw_waiting_world() -> void:
	for y in range(22):
		for x in range(32):
			var color := Color("294a40") if (x + y) % 3 else Color("315549")
			_draw_tile(x, y, color, false)
	draw_string(ThemeDB.fallback_font, Vector2(650, 450), "Waiting for Python world server...", HORIZONTAL_ALIGNMENT_LEFT, -1, 22, Color("b8d8c5"))
