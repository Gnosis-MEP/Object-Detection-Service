import abc


class VEKG(abc.ABC):
    """
    Base Gnosis Graph class
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def add_node(self, node_id, label, properties=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def add_edge(self, src_node, relation, dest_node, properties=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def nodes(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def edges(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def execute_query(self, query, parser=None):
        raise NotImplementedError()

    @abc.abstractmethod
    def commit(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def drop(self):
        raise NotImplementedError()