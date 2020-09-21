#
# Pypher is a tiny library that focuses on building Cypher queries by constructing pure Python objects.
# Link: https://github.com/emehrkay/pypher
#
# This class returns the query information in the Cypher notation.
#

import json
from pypher import Pypher, create_statement
from pypher.builder import Params


class CypherQLBuilder():

    def __init__(self):
        super().__init__()
        self.cypher_q = Pypher(params=Params(prefix='VAR'))
        self.create_clauses()

    def create_clauses(self):
        self.COMMA = ','
        create_statement('OR', {'name': 'OR'})
        create_statement('AND', {'name': 'AND'})

    def write_MATCH(self, query_info):
        self.cypher_q = self.cypher_q.MATCH
        for index, object in enumerate(query_info['objects']):
            self.cypher_q = self.cypher_q.node(object['obj_ref'], labels=object['obj_class'].upper())
            if index != len(query_info['objects']) - 1:
                self.cypher_q.raw(self.COMMA)

    def write_SET(self, query_info, attribute_map):
        self.cypher_q = self.cypher_q.Set
        for index, object in enumerate(query_info['objects']):
            obj_ref = object['obj_ref']
            create_statement('obj_ref', {'name': obj_ref})
            for attr_name, attr_value in attribute_map.items():
                self.cypher_q = self.cypher_q.obj_ref.property(attr_name).operator('=', attr_value)
                if index != len(query_info['objects']) - 1:
                    self.cypher_q.raw(self.COMMA)

    def write_WHERE(self, query_info):
        if 'predicate' in query_info:
            predicate = query_info['predicate']
            self.cypher_q = self.cypher_q.WHERE
            while predicate is not None:

                obj_ref = predicate['expression']['obj_ref']
                create_statement('obj_ref', {'name': obj_ref})
                attr_name = predicate['expression']['attr_name']
                comparison_operator = predicate['expression']['comparison_operator']
                attr_value = predicate['expression']['attr_value']
                if attr_name.upper() == 'LABEL':
                    attr_value = attr_value.upper()
                self.cypher_q = self.cypher_q.obj_ref.property(attr_name).operator(comparison_operator, attr_value)

                rel_type = predicate['rel_type']
                if rel_type is not None and rel_type == 'OR':
                    self.cypher_q = self.cypher_q.OR
                elif rel_type is not None and rel_type == 'AND':
                    self.cypher_q = self.cypher_q.AND

                predicate = predicate['next_predicate']

    def write_RETURN(self, query_info):
        if 'filter_by' in query_info:
            self.cypher_q = self.cypher_q.RETURN(', '.join(query_info['filter_by']))

    def get_cypher(self):
        query_i = str(self.cypher_q)
        query_q = list(query_i)
        indexes_of_comma = [i for i, char in enumerate(query_i) if char == ',']

        for index in indexes_of_comma:
            if query_q[index - 1] == ' ':
                query_q[index - 1] = ''

        return ''.join(query_q)

    def build_select_query(self, query_info):

        self.write_MATCH(query_info)
        self.write_WHERE(query_info)
        self.write_RETURN(query_info)

        return self.get_cypher(), dict(self.cypher_q.bound_params)

    def build_update_query(self, query_info, attribute_map):

        self.write_MATCH(query_info)
        self.write_WHERE(query_info)
        self.write_SET(query_info, attribute_map)
        self.write_RETURN(query_info)

        return self.get_cypher(), dict(self.cypher_q.bound_params)


if __name__ == '__main__':
    query = '{"query_name": "testQuery", "output_type": "ANN_IMAGE", "content_services": ["objectDetection", "ColorDetection"], "objects": [{"obj_ref": "p", "obj_class": "Person"}, {"obj_ref": "c", "obj_class": "Car"}], "predicate": {"expression": {"obj_ref": "p", "attr_name": "label", "comparison_operator": "=", "attr_value": "Person"}, "rel_type": "AND", "next_predicate": {"expression": {"obj_ref": "c", "attr_name": "label", "comparison_operator": "=", "attr_value": "Car"}, "rel_type": null, "next_predicate": null}}, "publisher_id": "pub01", "windows_info": {"window_type": "TUMBLING_COUNT_WINDOW", "window_length": "5"}, "qos": [{"metric_name": "CONFIDENCE", "comparison_operator": ">", "metric_value": "70"}], "filter_by": ["*"]}'
    query_info = json.loads(query)
    cypher = CypherQLBuilder()
    cypher_q, params = cypher.build_update_query(query_info, {'is_matched': 'True'})
    print(cypher_q)
    print(params)
