import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import sys
import tty
import termios

ROS_TOPIC = "unitree_key_cmd"

# ─── Aanpasbare test berichten ────────────────────────────────────────────────

TEST_MESSAGES = {
    "a": {"heading": 1,             "frame_id": 1},   # rechtdoor
    "z": {"heading": 2,             "frame_id": 1},   # licht links
    "e": {"heading": 3,            "frame_id": 1},   # licht rechts
    "q": {"heading": 1.57,            "frame_id": 1},   # scherp links
    "d": {"heading": -1,           "frame_id": 1},   # scherp rechts
    "w": {"heading": 6,            "frame_id": 1},   # max links
}

# ─────────────────────────────────────────────────────────────────────────────


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def print_controls():
    print("\n================= TEST PUBLISHER =================")
    for key, msg in TEST_MESSAGES.items():
        print(f"  {key}  -->  heading={msg['heading']}")
    print(f"  ctrl+c  -->  afsluiten")
    print("==================================================\n")


class TestPublisher(Node):

    def __init__(self):
        super().__init__('unitree_test_publisher')
        self.publisher_ = self.create_publisher(String, ROS_TOPIC, 10)
        self.frame_counter = 0

    def send(self, payload: dict):
        self.frame_counter += 1
        payload["frame_id"] = self.frame_counter
        msg = String()
        msg.data = json.dumps(payload)
        self.publisher_.publish(msg)
        self.get_logger().info(f"Sent: {payload}")


def main():
    rclpy.init()
    node = TestPublisher()
    print_controls()

    while rclpy.ok():
        try:
            k = get_key().lower()
        except Exception:
            break

        if k == "\x03":  # Ctrl+C
            break
        elif k in TEST_MESSAGES:
            node.send(dict(TEST_MESSAGES[k]))  # copy zodat origineel niet wordt aangepast
        else:
            print(f"Onbekende key: '{k}'")


if __name__ == '__main__':
    main()
