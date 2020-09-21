from .base import VEKG
from redisgraph import Node, Edge, Graph


class RedisGraph(VEKG):

    def __init__(self, graph, fetch_type):
        super().__init__()
        self.graph = graph
        self.nodes = self.graph.nodes
        self.edges = self.graph.edges
        if 'eager' == fetch_type:
            self.retrieve_all_nodes_and_edges()

    def add_node(self, node_id, label, properties=None):
        if properties is None:
            properties = {}
        node = Node(node_id=node_id, label=label, properties=properties)
        self.graph.add_node(node)

    def add_edge(self, src_node, dest_node, relation=None, properties=None):
        if properties is None:
            properties = {}
        if relation is None:
            relation = ''
        edge = Edge(src_node=src_node, relation=relation, dest_node=dest_node, properties=properties)
        self.graph.add_edge(edge)

    def nodes(self):
        return self.graph.nodes

    def edges(self):
        return self.graph.edges

    def retrieve_all_nodes_and_edges(self):
        result_set = self.execute_query('MATCH (n) RETURN n')
        for result in result_set:
            for node in result:
                self.graph.add_node(node)

        result_set = self.execute_query('MATCH (n)-[r]->(m) RETURN n,r,m')
        for result in result_set:
            for edge in result:
                self.graph.add_edge(edge)

    def execute_query(self, query, params=None):
        if params is None:
            params = dict()
        result = self.graph.query(query, params=params)
        result.pretty_print()
        return result.result_set

    def get_unique_result(self, result_set):
        result_map = dict()
        for result in result_set:
            for node in result:
                result_map[node.id] = node
        return result_map

    def commit(self):
        self.graph.flush()

    def drop(self):
        self.graph.delete()


class RedisGraphEngine():

    def __init__(self, redis_conn):
        self.redis_conn = redis_conn

    def get_graph_instance(self, graph_id, fetch_type="lazy"):
        redis_graph = Graph(graph_id, self.redis_conn)
        return RedisGraph(redis_graph, fetch_type)