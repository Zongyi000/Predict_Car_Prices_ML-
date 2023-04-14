import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_tree


def main():

    # Visualize XGBoost tree
    model = joblib.load('xgb_model.joblib')
    print(model)
    plot_tree(model, fmap='', num_trees=0, ax=None)
    plt.show()

    # Store the tree file
    graph = xgb.to_graphviz(model, num_trees=0, **{'size': str(10)})
    graph.render(filename='xgb.dot')


if __name__ == "__main__":
    main()
