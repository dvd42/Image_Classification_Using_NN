# A library and utility for drawing ONNX nets. Most of this implementation has
# been borrowed from the caffe2 implementation
# https://github.com/caffe2/caffe2/blob/master/caffe2/python/net_drawer.py
#
# The script takes two required arguments:
#   -input: a path to a serialized ModelProto .pb file.
#   -output: a path to write a dot file representation of the graph
#
# Given this dot file representation, you can-for example-export this to svg
# with the graphviz `dot` utility, like so:
#
#   $ dot -Tsvg my_output.dot -o my_output.svg
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
from collections import defaultdict
import json
from onnx import ModelProto
import pydot


OP_STYLE = {
    'shape': 'box',
    'color': '#00bfff',
    'style': 'filled',
    'fontcolor': '#000000'

}

BLOB_STYLE = {'shape': 'circle','style':'filled','fillcolor': 'orange',}


def _escape_label(name):
    # json.dumps is poor man's escaping
    return json.dumps(name)


def _form_and_sanitize_docstring(s):
    url = 'javascript:alert('
    url += _escape_label(s).replace('"', '\'').replace('<', '').replace('>', '')
    url += ')'
    return url


def GetOpNodeProducer(embed_docstring=False, **kwargs):
    def ReallyGetOpNode(op, op_id):
        if op.name:
            node_name = '%s/%s (op#%d)' % (op.name, op.op_type, op_id)
        else:
            node_name = '%s (op#%d)' % (op.op_type, op_id)
        for i, input in enumerate(op.input):
            node_name += '\n input' + str(i) + ' ' + input
        for i, output in enumerate(op.output):
            node_name += '\n output' + str(i) + ' ' + output
        node = pydot.Node(node_name, **kwargs)
        if embed_docstring:
            url = _form_and_sanitize_docstring(op.doc_string)
            node.set_URL(url)
        return node
    return ReallyGetOpNode


def GetPydotGraph(
    graph,
    name=None,
    rankdir='LR',
    node_producer=None,
    embed_docstring=False,
    bgcolor='#cdc9c9',
):
    if node_producer is None:
        node_producer = GetOpNodeProducer(embed_docstring=embed_docstring, **OP_STYLE)


    pydot_graph = pydot.Dot(name, rankdir=rankdir,bgcolor=bgcolor)
    pydot_nodes = {}
    pydot_node_counts = defaultdict(int)
    for op_id, op in enumerate(graph.node):
        print(op_id,op)
        op_node = node_producer(op, op_id)
        pydot_graph.add_node(op_node)
        for input_name in op.input:
            if input_name not in pydot_nodes:
                input_node = pydot.Node(
                    _escape_label(
                        input_name + str(pydot_node_counts[input_name])),
                    label=_escape_label(input_name),
                    **BLOB_STYLE
                )
                pydot_nodes[input_name] = input_node
            else:
                input_node = pydot_nodes[input_name]
            pydot_graph.add_node(input_node)
            pydot_graph.add_edge(pydot.Edge(input_node, op_node))
        for output_name in op.output:
            if output_name in pydot_nodes:
                pydot_node_counts[output_name] += 1
            output_node = pydot.Node(
                _escape_label(
                    output_name + str(pydot_node_counts[output_name])),
                label=_escape_label(output_name),
                **BLOB_STYLE
            )
            pydot_nodes[output_name] = output_node
            pydot_graph.add_node(output_node)
            pydot_graph.add_edge(pydot.Edge(op_node, output_node))
    return pydot_graph


def main():
    parser = argparse.ArgumentParser(description="ONNX net drawer")
    parser.add_argument(
        "--input",
        type=str, required=True,
        help="The input protobuf file.",
    )
    parser.add_argument(
        "--output",
        type=str, required=True,
        help="The output protobuf file.",
    )
    parser.add_argument(
        "--rankdir", type=str, default='LR',
        help="The rank direction of the pydot graph.",
    )
    parser.add_argument(
        "--embed_docstring", action="store_true",
        help="Embed docstring as javascript alert. Useful for SVG format.",
    )
    args = parser.parse_args()
    model = ModelProto()
    with open(args.input, 'rb') as fid:
        content = fid.read()
        model.ParseFromString(content)
    pydot_graph = GetPydotGraph(
        model.graph,
        name=model.graph.name,
        rankdir=args.rankdir,
        node_producer=GetOpNodeProducer(
            embed_docstring=args.embed_docstring,
            **OP_STYLE
        ),
    )
    pydot_graph.write_dot(args.output)


if __name__ == '__main__':
    main()
