import ray


@ray.remote
class SyncServer(object):
    def __init__(self):
        self.node_stop_flag: bool = False

    def should_stop(self):
        return self.node_stop_flag

    def stop_the_world(self):
        self.node_stop_flag = True


def setup_sync_server():
    options = {
        "name": "SyncServer",
        "lifetime": "detached"
    }
    print("Sync server set!")
    return SyncServer.options(**options).remote()
