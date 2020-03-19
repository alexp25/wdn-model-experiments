from sklearn import tree
import pydot
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
import pandas as pd


def disp_boston(dataset, samples = 0):
    # Print the value of the dataset to understand what it contains.
    print(dataset.keys())
    # Find out more about the features use dataset.DESCR
    print(dataset.DESCR)
    print(dataset.feature_names)
    # create a data frame and show the data in tabular form

    data = dataset.data
    target_data = dataset.target

    if samples != 0:
        data = dataset.data[:samples]
        target_data = dataset.target[:samples]

    df = pd.DataFrame(data, columns=dataset.feature_names)
    print(df.head())
    # include the target data

    df['MEDV'] = target_data
    print(df.head())
    return df


def print_decision_tree(mytree, features, offset_unit='    '):
    '''Plots textual representation of rules of a decision tree
    tree: scikit-learn representation of tree
    feature_names: list of feature names. They are set to f1, f2, f3,... if not specified
    offset_unit: a string of offset of the conditional block'''

    print("\ndecision tree:\n")

    left = mytree.tree_.children_left
    right = mytree.tree_.children_right
    threshold = mytree.tree_.threshold
    value = mytree.tree_.value
    if features is None:
        features = ['f%d' % i for i in mytree.tree_.feature]
    else:
        features = [features[i] for i in mytree.tree_.feature]

    def recurse(left, right, threshold, features, node, depth=0):
        offset = offset_unit * depth
        if threshold[node] != -2:
            print(offset + "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node], depth + 1)
            print(offset + "} else {")
            if right[node] != -1:
                recurse(left, right, threshold, features, right[node], depth + 1)
            print(offset + "}")
        else:
            v = value[node][0]

            if v[0] < v[1]:
                resp = "True"
            else:
                resp = "False"

            print(offset + "return " + str(v[0]) + "/" + str(v[1]) + " => " + resp)

    recurse(left, right, threshold, features, 0, 0)


def print_tree_graph(mytree, features):
    # dot_data = StringIO()
    # tree.export_graphviz(mytree, out_file=dot_data, feature_names=features)
    # graph = pydot.graph_from_dot_data(dot_data.getvalue())
    # graph = graph[0]
    # print(graph)
    tree.export_graphviz(mytree, out_file="mytree.dot", feature_names=features)
    # graph.export_png("test.png")