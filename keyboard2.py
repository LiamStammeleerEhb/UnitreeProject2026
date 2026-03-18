import asyncio
import logging
import json
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

MOVE_SPEED = 0.5
TURN_SPEED = 1

def print_controls():
    print("""
================= CONTROLS =================
z : vooruit
s : achteruit
a : draaien links
e : draaien rechts
q : zijwaarts links
d : zijwaarts rechts
o : opstaan
l : neerliggen
h : hello
x : stretch
p : print controls
(ctrl + c) * 2 : exit
============================================
""")

async def send_move(conn, x=0, y=0, z=0):
    await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["SPORT_MOD"],
        {
            "api_id": SPORT_CMD["Move"],
            "parameter": {"x": x, "y": y, "z": z}
        }
    )

def get_key():
    """Read a single keypress from stdin (works over SSH)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def keyboard_listener(conn, loop, stop_event):
    """Blocking keyboard loop running in a separate thread."""
    while not stop_event.is_set():
        try:
            k = get_key().lower()
        except Exception:
            break

        # Ctrl+C
        if k == "\x03":
            print("\nExit requested")
            stop_event.set()
            break

        if k == "z":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, x=MOVE_SPEED), loop)

        elif k == "s":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, x=-MOVE_SPEED), loop)

        elif k == "a":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, z=TURN_SPEED), loop)

        elif k == "e":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, z=-TURN_SPEED), loop)

        elif k == "q":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, y=MOVE_SPEED), loop)

        elif k == "d":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, y=-MOVE_SPEED), loop)

        elif k == " ":
            asyncio.run_coroutine_threadsafe(
                send_move(conn, x=0), loop)

        elif k == "h":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["Hello"]}
                ), loop)

        elif k == "x":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["Stretch"], "parameter": {"data": False}}
                ), loop)

        elif k == "l":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["StandDown"]}
                ), loop)

        elif k == "o":
            asyncio.run_coroutine_threadsafe(
                conn.datachannel.pub_sub.publish_request_new(
                    RTC_TOPIC["SPORT_MOD"],
                    {"api_id": SPORT_CMD["RecoveryStand"], "parameter": {"data": False}}
                ), loop)

        elif k == "p":
            print_controls()


async def main():
    conn = UnitreeWebRTCConnection(
        WebRTCConnectionMethod.LocalSTA,
        ip="10.2.172.247"
    )

    await conn.connect()

    # Ensure normal mode
    response = await conn.datachannel.pub_sub.publish_request_new(
        RTC_TOPIC["MOTION_SWITCHER"],
        {"api_id": 1001}
    )
    data = json.loads(response['data']['data'])
    if data["name"] != "normal":
        await conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {"api_id": 1002, "parameter": {"name": "normal"}}
        )
        await asyncio.sleep(5)

    print_controls()

    loop = asyncio.get_running_loop()
    stop_event = threading.Event()

    # Run keyboard listener in a background thread so it doesn't block the async loop
    kb_thread = threading.Thread(
        target=keyboard_listener,
        args=(conn, loop, stop_event),
        daemon=True
    )
    kb_thread.start()

    # Wait until stop_event is set (Ctrl+C in the terminal)
    while not stop_event.is_set():
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted")
        sys.exit(0)
