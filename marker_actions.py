{
  "markers": [
    {
      "marker_id": 0,
      "trigger_distance": 1.5,
      "actions": [
        { "type": "stop_walk" },
        { "type": "turn_degrees", "degrees": -90 },
        { "type": "set_walk_speed", "speed": 0.5 }
      ]
    },
    {
      "marker_id": 1,
      "trigger_distance": 1.0,
      "actions": [
        { "type": "stop_walk" },
        { "type": "turn_degrees", "degrees": 90 },
        { "type": "pause", "seconds": 2.0 },
        { "type": "set_inactive" }
      ]
    },
    {
      "marker_id": 2,
      "trigger_distance": 0.8,
      "actions": [
        { "type": "stop_walk" },
        { "type": "turn_degrees", "degrees": -90 },
        { "type": "pause", "seconds": 1.5 },
        { "type": "set_walk_speed", "speed": 0.3 }
      ]
    },
    {
      "marker_id": 3,
      "trigger_distance": 3,
      "actions": [
        { "type": "set_walk_speed", "speed": 0.7 }
      ]
    },
    {
      "marker_id": 4,
      "trigger_distance": 3,
      "actions": [
        { "type": "set_walk_speed", "speed": 0.3 },
        { "type": "pause", "seconds": 0.5 },
        { "type": "stop_walk" },
        { "type": "pause", "seconds": 0.5 },
        { "type": "turn_degrees", "degrees": 180 },
        { "type": "pause", "seconds": 0.5 },
        { "type": "set_walk_speed", "speed": 0.6 }
      ]
    }
  ]
}
