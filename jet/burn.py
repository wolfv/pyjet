import pygraphviz as pyg
import networkx as nx
import os
from jet.config import group_class, group_func


def draw(graph, outputs=[], name='graph'):
    dg = pyg.AGraph(directed=True)
    output_names = [out.name for out in outputs]
    class_set = set()
    subgraph_dict = {}
    for node in graph.nodes():
        if node.op == 'Const':
            dg.add_node(repr(node), fillcolor='Coral', style='filled')
        elif node.op == 'Variable':
            dg.add_node(repr(node), fillcolor='lightblue', style='filled')        
        elif node.op == 'Placeholder':
            dg.add_node(repr(node), shape='invhouse', style='filled,bold')
        elif node.name in output_names:
            dg.add_node(repr(node), shape='house', style='filled,bold')
        else:
            dg.add_node(repr(node), fillcolor='white', style='filled')
        if group_class or group_func:
            class_set.add(node.caller)

    for edge in graph.edges():
        data = graph.get_edge_data(edge[0], edge[1])
        if data and data.get('edge_type') == 'helper':
            dg.add_edge(edge[0], edge[1],
                            style='dashed', constraint='false', color='grey')
        else:
            dg.add_edge(edge[0], edge[1])

    # group op-nodes
    if group_class:
        class_name = ''
        for class_info in class_set:
            if class_name != class_info[0]:
                class_name = class_info[0]
                grouped_class = [n for n, d in graph.node.items()
                                                if n.caller[0] == class_info[0]]
                sub_graph = dg.subgraph(grouped_class,
                                        name='cluster_class_' + class_info[1],
                                        label=class_info[1],
                                        color='azure3',
                                        fillcolor='aliceblue',
                                        style='filled')
            grouped_func = [n for n, d in graph.node.items() if n.caller == class_info]
            if group_func:
                sub_graph.add_subgraph(grouped_func,
                                       name='cluster_func_'+ class_info[2],
                                       label=class_info[2],
                                       color='grey75',
                                       fillcolor='grey94',
                                       style='filled')
    elif group_func:
        for class_info in class_set:
            grouped_func = [n for n, d in graph.node.items() if n.caller == class_info]
            if group_func:
                dg.add_subgraph(grouped_func,
                                name='cluster_func_'+ class_info[2],
                                label=class_info[2],
                                color='grey75',
                                fillcolor='grey94',
                                style='filled')


    if not os.path.exists(os.path.dirname('jet_generated/jet_graph/')):
        try:
            os.makedirs(os.path.dirname('jet_generated/jet_graph/'))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    dg.write('jet_generated/jet_graph/' + name + '.dot')
    dg.draw('jet_generated/jet_graph/' + name + '.ps', prog='dot')
