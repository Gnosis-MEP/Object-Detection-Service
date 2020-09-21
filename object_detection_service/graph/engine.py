from walrus import Database

from .redis_graph import RedisGraphEngine


class GraphEngineFactory():

    def __init__(self, host='localhost', port='6379'):
        super().__init__()
        self.host = host
        self.port = port

    def create(self, engine_type):
        if engine_type == 'redis_graph':
            return RedisGraphEngine(Database(host=self.host, port=self.port))