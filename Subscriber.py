import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
from std_msgs.msg import String

import asyncio
import json
import logging
import math
import os
import sys
import tty
import termios
import threading

from unitree_webrtc_connect.webrtc_driver import (
    UnitreeWebRTCConnection,
    WebRTCConnectionMethod
)
from unitree_webrtc_connect.constants import RTC_TOPIC, SPORT_CMD

logging.basicConfig(level=logging.FATAL)

ROBOT_IP            = "unitree.local"
MOVE_SPEED          = 0.5
MAX_TURN_SPEED      = 2.0
ROS_TOPIC           = "unitree_key_cmd"
MARKER_CONFIG_FILE  = os.path.join(os.path.dirname(__file__), "marker_actions.json")

CMD_RATE_HZ    = 10
CMD_INTERVAL   = 1.0 / CMD_RATE_HZ

KEY_START      = "w"
KEY_STOP       = "x"
KEY_SPEED_UP   = "z"
KEY_SPEED_DOWN = "s"
KEY_SPEED_ZERO = "a"
KEY_STANDUP    = "e"
KEY_LAYDOWN    = "r"

WALK_STEP      = 0.1
Set_walk = 0.0


def load_marker_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] Marker config not found at {path}, no marker actions will run.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    config = {}
    for entry in raw.get("markers", []):
        mid = int(entry["marker_id"])
        config[mid] = {
            "trigger_distance": float(entry.get("trigger_distance", 1.0)),
            "actions": entry.get("actions", [])
        }

    print(f"[INFO] Loaded marker config: {len(config)} marker(s) configured.")
    return config


async def send_move(conn, x=0.0, y=0.0, z=0.0):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": x, "y": y, "z": z}
        }
    )


async def send_rotate(conn, degrees: float, speed: float = 1.0):
    radians = math.radians(degrees)
    direction = 1.0 if radians >= 0 else -1.0
    duration = abs(radians) / max(0.001, speed)
    z_cmd = direction * speed

    steps = max(1, int(duration / CMD_INTERVAL))
    for _ in range(steps):
        await send_move(conn, x=0.0, z=z_cmd)
        await asyncio.sleep(CMD_INTERVAL)
    await send_move(conn, x=0.0, z=0.0)


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def print_status(active: bool):
    status = "ACTIEF" if active else "GESTOPT"
    print(f"\n[STATUS] Robot: {status} | Set_walk={Set_walk:.2f}")
    print(
        f"  {KEY_START}=start  {KEY_STOP}=stop  "
        f"  {KEY_SPEED_UP}=+speed  {KEY_SPEED_DOWN}=-speed  {KEY_SPEED_ZERO}=speed→0  "
        f"  {KEY_STANDUP}=opstaan  {KEY_LAYDOWN}=neerliggen  "
        f"  ctrl+c=afsluiten\n"
    )


class UnitreeHeadingSubscriber(Node):
    def __init__(self, conn, loop, marker_config: dict):
        super().__init__("unitree_heading_subscriber")
        self.conn = conn
        self.loop = loop
        self.marker_config = marker_config

        self.active = False
        self.marker_busy = False
        self.triggered_markers: set[int] = set()
        self.last_cmd_time = 0.0

        self.get_logger().set_level(LoggingSeverity.WARN)

        self.subscription = self.create_subscription(
            String, ROS_TOPIC, self.heading_callback, 10
        )
        self.get_logger().warn(f"Subscriber active on '{ROS_TOPIC}'")

    def set_active(self, state: bool):
        self.active = state
        self.get_logger().info(f"Robot controle {'GESTART' if state else 'GESTOPT'}")

    def heading_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Ongeldige JSON: {e}")
            return

        self._process_markers_from_data(data)

        if not self.active or self.marker_busy:
            return

        now = self.get_clock().now().nanoseconds / 1e9
        if (now - self.last_cmd_time) < CMD_INTERVAL:
            return
        self.last_cmd_time = now

        heading = data.get("heading")
        if heading is None:
            return

        heading = max(-math.pi, min(math.pi, float(heading)))
        zz = -(heading / math.pi) * MAX_TURN_SPEED

        asyncio.run_coroutine_threadsafe(
            send_move(self.conn, x=Set_walk, z=zz),
            self.loop
        )

    def _process_markers_from_data(self, data: dict):
        markers = data.get("markers", [])
        if not isinstance(markers, list):
            return

        for marker in markers:
            if not isinstance(marker, dict):
                continue

            mid = marker.get("marker_id", marker.get("id"))
            distance = marker.get("marker_distance", marker.get("distance"))
            if mid is None or distance is None:
                continue

            try:
                mid = int(mid)
                distance = float(distance)
            except (TypeError, ValueError):
                continue

            cfg = self.marker_config.get(mid)
            if cfg is None:
                continue

            trigger_dist = float(cfg.get("trigger_distance", 1.0))

            if distance <= trigger_dist and mid not in self.triggered_markers:
                self.triggered_markers.add(mid)
                asyncio.run_coroutine_threadsafe(
                    self._execute_marker_actions(mid, cfg.get("actions", [])),
                    self.loop
                )
                break
            elif distance > trigger_dist and mid in self.triggered_markers:
                self.triggered_markers.discard(mid)

    async def _execute_marker_actions(self, marker_id: int, actions: list):
        global Set_walk

        self.marker_busy = True
        self.get_logger().warn(
            f"[MARKER {marker_id}] Starting sequence ({len(actions)} actions)"
        )
        await send_move(self.conn, x=0.0, y=0.0, z=0.0)

        for i, action in enumerate(actions):
            atype = action.get("type")
            self.get_logger().warn(
                f"[MARKER {marker_id}] Action {i + 1}/{len(actions)}: {atype}"
            )

            if atype == "stop_walk":
                Set_walk = 0.0
                await send_move(self.conn, x=0.0, z=0.0)

            elif atype == "set_walk_speed":
                Set_walk = max(0.0, min(1.0, float(action.get("speed", 0.0))))
                print_status(self.active)

            elif atype == "turn_degrees":
                await send_rotate(
                    self.conn,
                    degrees=float(action.get("degrees", 0.0)),
                    speed=float(action.get("speed", 1.0))
                )

            elif atype == "pause":
                await asyncio.sleep(float(action.get("seconds", 1.0)))

            elif atype == "set_inactive":
                self.active = False
                self.get_logger().warn(f"[MARKER {marker_id}] Robot set INACTIVE")
                print_status(False)

            elif atype == "give_paw":
                await self.conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["Hello"]}
                )

            else:
                self.get_logger().warn(
                    f"[MARKER {marker_id}] Unknown action '{atype}', skipping."
                )

        self.get_logger().warn(f"[MARKER {marker_id}] Sequence complete.")
        self.marker_busy = False


def keyboard_listener(node, conn, loop, stop_event):
    global Set_walk
    print_status(node.active)

    while not stop_event.is_set():
        try:
            k = get_key().lower()
        except Exception:
            break

        if k == "\x03":
            stop_event.set()
            break

        elif k == KEY_START:
            if not node.marker_busy:
                node.set_active(True)
                print_status(True)
            else:
                print("[WARN] Cannot start: marker action running.")

        elif k == KEY_STOP:
            node.set_active(False)
            Set_walk = 0.0
            asyncio.run_coroutine_threadsafe(send_move(conn, x=0.0, z=0.0), loop)
            print_status(False)

        elif k == KEY_SPEED_UP:
            if not node.marker_busy:
                Set_walk = min(1.0, Set_walk + WALK_STEP)
                print_status(node.active)
            else:
                print("[WARN] Cannot change speed: marker action running.")

        elif k == KEY_SPEED_DOWN:
            if not node.marker_busy:
                Set_walk = max(0.0, Set_walk - WALK_STEP)
                print_status(node.active)
            else:
                print("[WARN] Cannot change speed: marker action running.")

        elif k == KEY_SPEED_ZERO:
            Set_walk = 0.0
            asyncio.run_coroutine_threadsafe(send_move(conn, x=0.0, z=0.0), loop)
            print_status(node.active)

        elif k == KEY_STANDUP:
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["RecoveryStand"], "parameter": {"data": False}}
                ),
                loop
            )
            print("[CMD] Opstaan")

        elif k == KEY_LAYDOWN:
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["StandDown"]}
                ),
                loop
            )
            print("[CMD] Neerliggen")


async def main_async():
    marker_config = load_marker_config(MARKER_CONFIG_FILE)

    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP
    )
    await conn.connect()

    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"],
        {"api_id": 1001}
    )
    mode_data = json.loads(response["data"]["data"])
    if mode_data["name"] != "normal":
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": "normal"}}
        )
        await asyncio.sleep(5)

    rclpy.init()
    loop = asyncio.get_running_loop()
    node = UnitreeHeadingSubscriber(conn, loop, marker_config)

    stop_event = threading.Event()
    kb_thread = threading.Thread(
        target=keyboard_listener,
        args=(node, conn, loop, stop_event),
        daemon=True
    )
    kb_thread.start()

    try:
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0)
            await asyncio.sleep(0.01)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nShutdown requested")


if __name__ == "__main__":
    main()
