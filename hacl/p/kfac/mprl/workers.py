import time
import jactorch
import threading
import queue
import ray


@ray.remote
class DataWorker(object):
    def __init__(self, i, args, data_store, worker_fn):
        self.index = i
        self.args = args
        self.data_store = data_store
        self.worker_fn = worker_fn

        self.model_parameters = None
        self.mutex = threading.Lock()

        self.thread = threading.Thread(target=self.worker_fn, daemon=True, args=(
            self.index, args, self.maybe_update_model, self.push_data
        ))
        self.thread.start()

    def set_model_parameters(self, params):
        with self.mutex:
            self.model_parameters = params

    def get_model_parameters(self):
        with self.mutex:
            param = self.model_parameters
            self.model_parameters = None
            return param

    def maybe_update_model(self, model):
        param = self.get_model_parameters()
        if param == 'exit':
            return False
        if param is not None:
            print('Worker {}: updating the model...'.format(self.index))
            jactorch.load_state_dict(model, param)
        return True

    def push_data(self, data):
        ray.get(self.data_store.push.remote(data))


@ray.remote
class DataStore(object):
    def __init__(self):
        self.queue = queue.Queue(64)

    def push(self, data):
        while True:
            try:
                self.queue.put_nowait(data)
                break
            except:
                pass
            time.sleep(0.5)

        return True

    def pop(self):
        try:
            return self.queue.get_nowait()
        except:
            return None

