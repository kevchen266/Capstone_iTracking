import threading
import time
from .queue_manager import q as data_queue
import logging
from .db import insert_to_cdb
import base64

logger = logging.getLogger(__name__)

def process_calibration():
    '''Process frames from queue'''
    while True:
        if not data_queue.empty():
            data = data_queue.get()  # Dequeue a frame
            images_bytes = bytes((int + 256) if int < 0 else int for int in data[1])
            try:
                # print(f"Inserting frame @ {data[0]['coordinates']}")
                insert_to_cdb(data[0]["coordinates"], images_bytes)
            except Exception as e:
                print()
                # logger.error(f"Error processing frame: {e}")
            finally:
                data_queue.task_done()  # Mark the frame as processed
        else:
            time.sleep(0.1)  # Avoid busy-waiting when the queue is empty


def start2():
    threading.Thread(target=process_calibration, daemon=True).start()
