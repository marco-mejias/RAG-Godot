extends CharacterBody2D

const SPEED = 300.0
const JUMP_VELOCITY = -400.0

@onready var sprite = $Sprite
var health_label: Label


func _ready() -> void:
	health_label = get_node("../HUD/HealthLabel")


func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity += get_gravity() * delta

	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	var direction := Input.get_axis("ui_left", "ui_right")
	velocity.x = direction * SPEED if direction else move_toward(velocity.x, 0, SPEED)

	sprite.flip_h = direction < 0
	health_label.text = "HP: 100"
	move_and_slide()
