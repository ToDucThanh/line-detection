# import argparse
# import sys

# import cv2

# # import urllib.parse
# import redis

# """
# Capture frames from Camera and save to Redis Streams
# Example: python3 edge-camera.py -u redis://redis:6379 --fps 6 --rotate-90-clockwise true
# """
# if __name__ == "__main__":
#     """
#     Parse arguments
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--host", help="Redis host", type=str, default="localhost")
#     parser.add_argument("--port", help="Redis port", type=int, default=6379)
#     parser.add_argument(
#         "-o", "--output", help="Output stream key name", type=str, default="camera:0"
#     )
#     parser.add_argument("--fmt", help="Frame storage format", type=str, default=".jpg")
#     parser.add_argument(
#         "--fps", help="Frames per second (webcam)", type=float, default=1.0
#     )
#     parser.add_argument(
#         "--maxlen", help="Maximum length of output stream", type=int, default=1000
#     )
#     parser.add_argument("--width", help="Width of the frame", type=int, default=640)
#     parser.add_argument("--height", help="Height of the frame", type=int, default=480)
#     parser.add_argument("--rotate-90-clockwise", help="Angle to rotate", type=bool)
#     args = parser.parse_args()

#     """
#     Set up Redis connection
#     """
#     # url = urllib.parse.urlparse(args.url)
#     conn = redis.Redis(host=args.host, port=args.port)
#     if not conn.ping():
#         raise Exception("Redis unavailable")
#     print("Connected to Redis: {}:{}".format(args.host, args.port))

#     """
#     Open the camera device at the ID 0
#     """
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         raise Exception("Could not open video device")

#     """
#     Set camera resolution and FPS
#     """
#     cap.set(cv2.CAP_PROP_FPS, args.fps)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

#     """
#     Capture frames
#     """
#     while True:
#         try:
#             ret, frame = cap.read()

#             """
#             Rotate 90 degree clockwise if required
#             """
#             if args.rotate_90_clockwise:
#                 frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

#             """
#             Encode and add frame to Redis Stream
#             """
#             _, data = cv2.imencode(args.fmt, frame)
#             img = data.tobytes()
#             id = conn.execute_command(
#                 "xadd", args.output, "MAXLEN", "~", args.maxlen, "*", "img", img
#             )
#             print("id: {}, size: {}".format(id, len(img)))

#         except Exception as e:
#             print(f"Error: {e}")
#             print("Releasing the capture and exiting...")

#             """
#             Release the capture
#             """
#             cap.release()
#             cv2.destroyAllWindows()
#             sys.exit()


from typing import Union

import cv2
import numpy as np
import redis

from .redis_database import conn


class CameraPipeline:
    def __init__(
        self,
        conn: redis.client.Redis = conn,
        fps: int = 15,
        width: int = 640,
        height: int = 480,
        fmt: str = ".jpg",
    ):
        self.fps = fps
        self.conn = conn
        self.width = width
        self.height = height
        self.fmt = fmt

    def push_frame_data_to_redis(
        frame: np.ndarray,
        camera_id: Union[str, int],
    ):
        """Run predict for a frame"""
        try:
            frame_json = {
                "camera_id": camera_id,
                "frame_timestamp_mili": 1000,
                "frame_starting_time_mili": 1000,
            }

            # Save frame to Redis
            _, data = cv2.imencode(cfg["image"]["fmt"], frame)
            msg = {"json": json.dumps(frame_json), "image": data.tobytes()}

            # create frame redis_id in micro second
            frame_redis_id = str(int(frame_timestamp_mili * 1000))
            # frame_redis_id_ = conn.xadd(f'camera:{camera_id}', msg, id=frame_redis_id, maxlen=10000)
            frame_redis_id_ = conn.xadd(camera_id, msg, id=frame_redis_id, maxlen=10000)
            # logger.debug(f'Camera: {camera_id} - Push frame to redis successfully')
            return frame_redis_id_
        except Exception as e:
            logger.debug("push frame data:::Error: {}".format(e))

    def read_camera_push_redis(
        self,
        camera_id: Union[str, int] = 0,
    ):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise Exception("Could not open video device")

        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.heigth)
        while True:
            # frame_starting_time_mili = time() * 1000
            ret, frame = cap.read()
            if not ret:
                continue
            _, data = cv2.imencode(self.fmt, frame)
            img = data.tobytes()
