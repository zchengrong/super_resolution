import threading
import uuid

id_lock = threading.Lock()


def generate_uuid():
    with id_lock:
        unique_id = str(uuid.uuid1())
    return unique_id
