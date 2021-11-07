# -*- coding: UTF-8 -*-

import random
from numpy import *

# defined the activate function
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# every node need to contains it's upstream unit and downstream unit
# use layer index and node index to specify the node position
class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        """
        the output is sum of every upstream's output times their weight,
        then do with the activate function
        """
        output = 0
        for conn in self.upstream:
            output += conn.upstream_node.output * conn.weight
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (
            self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(
            lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret +
                              '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(
            lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
                # print 'sample %d training finished' % d

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        for layer in self.layers:
            layer.dump()


class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec)
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return [(x + y) for [x, y] in binary]


def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b,
                        map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)
                            )
                        )


def gradient_check(network, sample_feature, sample_label):
    '''
    姊害妫€鏌?    network: 绁炵粡缃戠粶瀵硅薄
    sample_feature: 鏍锋湰鐨勭壒寰?    sample_label: 鏍锋湰鐨勬爣绛?    '''
    # 璁＄畻缃戠粶璇樊
    def network_error(vec1, vec2): return \
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    # 鑾峰彇缃戠粶鍦ㄥ綋鍓嶆牱鏈笅姣忎釜杩炴帴鐨勬搴?    network.get_gradient(sample_feature, sample_label)

    # 瀵规瘡涓潈閲嶅仛姊害妫€鏌?
    for conn in network.connections.connections:
        # 鑾峰彇鎸囧畾杩炴帴鐨勬搴?        actual_gradient = conn.get_gradient()

        # 澧炲姞涓€涓緢灏忕殑鍊硷紝璁＄畻缃戠粶鐨勮宸?        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # 鍑忓幓涓€涓緢灏忕殑鍊硷紝璁＄畻缃戠粶鐨勮宸?        conn.weight -= 2 * epsilon # 鍒氭墠鍔犺繃浜嗕竴娆★紝鍥犳杩欓噷闇€瑕佸噺鍘?鍊?        error2 = network_error(network.predict(sample_feature), sample_label)

        # 鏍规嵁寮?璁＄畻鏈熸湜鐨勬搴﹀€?        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 鎵撳嵃
        print 'expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient)


def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def test(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print '\ttestdata(%u)\tpredict(%u)' % (
        data, normalizer.denorm(predict_data))


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print 'correct_ratio: %.2f%%' % (correct / 256 * 100)


def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    correct_ratio(net)
