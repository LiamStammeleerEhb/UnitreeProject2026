import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import asyncio
import json
import logging
import math
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

# ---- Config Settings ----------------

ROBOT_IP       = "unitree.local"
MOVE_SPEED     = 0.5
MAX_TURN_SPEED = 1.0
ROS_TOPIC      = "unitree_key_cmd"

KEY_START      = "s"   # start moving 
KEY_STOP       = "x"   # Stop moving

CMD_RATE_HZ     = 10       # Max Commands/Second
CMD_INTERVAL    = 1.0 / CMD_RATE_HZ   # Calculated automaically DO NOT CHANGE

# --------------------------------------


async def send_move(conn, x=0.0, y=0.0, z=0.0):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": x, "y": y, "z": z}
        }
    )


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
    print(f"\n[STATUS] Robot controle: {status}")
    print(f"  '{KEY_START}' = start  |  '{KEY_STOP}' = stop  |  ctrl+c = afsluiten\n")


class UnitreeHeadingSubscriber(Node):

    def __init__(self, conn, loop):
        super().__init__('unitree_heading_subscriber')
        self.conn = conn
        self.loop = loop
        self.active = False
        self.last_cmd_time = 0.0  # timestamp van laatste verwerkt commando

        self.subscription = self.create_subscription(
            String,
            ROS_TOPIC,
            self.heading_callback,
            10
        )
        self.get_logger().info(f"Subscriber luistert op '{ROS_TOPIC}' (controle GESTOPT)")

    def set_active(self, state: bool):
        self.active = state
        status = "GESTART" if state else "GESTOPT"
        self.get_logger().info(f"Robot controle {status}")

    def heading_callback(self, msg):
        if not self.active:
            self.get_logger().debug("Bericht ontvangen maar controle is gestopt, wordt genegeerd.")
            return

        # Rate limiting: negeer bericht als het te snel na het vorige komt
        now = self.get_clock().now().nanoseconds / 1e9  # tijd in seconden
        if (now - self.last_cmd_time) < CMD_INTERVAL:
            self.get_logger().debug(f"Bericht genegeerd (rate limiting, max {CMD_RATE_HZ} Hz)")
            return
        self.last_cmd_time = now

        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Ongeldige JSON: {e}")
            return

        heading = data.get("heading")
        frame_id = data.get("frame_id", "unknown")

        if heading is None:
            self.get_logger().warn("Geen 'heading' veld gevonden in JSON.")
            return

        self.get_logger().info(f"frame_id={frame_id} | heading={heading} rad")

        heading = max(-math.pi, min(math.pi, heading))
        zz = (heading / math.pi) * MAX_TURN_SPEED

        asyncio.run_coroutine_threadsafe(
            send_move(self.conn, x=0.0, z=zz),
            self.loop
        )


def keyboard_listener(node, stop_event):
    """Loopt in een aparte thread, leest keyboard input."""
    print_status(node.active)
    while not stop_event.is_set():
        try:
            k = get_key().lower()
        except Exception:
            break

        if k == "\x03":  # Ctrl+C
            stop_event.set()
            break
        elif k == KEY_START:
            node.set_active(True)
            print_status(True)
        elif k == KEY_STOP:
            node.set_active(False)
            print_status(False)


async def ros_spin(node):
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0)
        await asyncio.sleep(0.01)


async def main_async():
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip=ROBOT_IP
    )
    await conn.connect()

    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"],
        {"api_id": 1001}
    )
    mode_data = json.loads(response['data']['data'])
    if mode_data["name"] != "normal":
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": "normal"}}
        )
        await asyncio.sleep(5)

    rclpy.init()
    loop = asyncio.get_running_loop()
    node = UnitreeHeadingSubscriber(conn, loop)

    stop_event = threading.Event()

    kb_thread = threading.Thread(
        target=keyboard_listener,
        args=(node, stop_event),
        daemon=True
    )
    kb_thread.start()

    try:
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0)
            await asyncio.sleep(0.01)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nShutdown requested")


if __name__ == '__main__':
    main()