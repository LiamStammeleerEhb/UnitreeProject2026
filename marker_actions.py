# Possible functions

#{ "type": "stop_walk" },
#{ "type": "turn_degrees", "degrees": 90 },
#{ "type": "pause", "seconds": 2.0 },
#{ "type": "set_inactive" },
#{ "type": "set_walk_speed", "speed": 0.5 }

{
  "markers": [
    {
      "marker_id": 0,
      "trigger_distance": 1.5,
      "actions": [
        { "type": "stop_walk" },
        { "type": "pause", "seconds": 0.2 },
        { "type": "turn_degrees", "degrees": -90 },
        { "type": "pause", "seconds": 0.2 },
        { "type": "set_walk_speed", "speed": 0.5 }
      ]
    },
    {
      "marker_id": 1,
      "trigger_distance": 3.0,
      "actions": [
        { "type": "set_walk_speed", "speed": 0.8 }
      ]
    },
    {
      "marker_id": 2,
      "trigger_distance": 4,
      "actions": [
        { "type": "set_walk_speed", "speed": 0.3 },
        { "type": "pause", "seconds": 0.2 },
        { "type": "stop_walk" },
        { "type": "pause", "seconds": 0.2 },
        { "type": "turn_degrees", "degrees": 180 },
        { "type": "pause", "seconds": 0.2 },
        { "type": "set_walk_speed", "speed": 0.6 }
      ]
    },
    {
      "marker_id": 3,
      "trigger_distance": 3,
      "actions": [
        { "type": "stop_walk" },
        { "type": "pause", "seconds": 0.2 },
        { "type": "turn_degrees", "degrees": 90 },
        { "type": "pause", "seconds": 0.2 },
        { "type": "set_walk_speed", "speed": 0.5 }
      ]
    },
    {
      "marker_id": 4,
      "trigger_distance": 2,
      "actions": [
        { "type": "set_walk_speed", "speed": 0.3 },
        { "type": "pause", "seconds": 0.2 },
        { "type": "stop_walk" },
        { "type": "set_inactive" },
      ]
    }
  ]
}
