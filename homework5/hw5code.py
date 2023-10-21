import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator


def find_best_split(feature_vector, target_vector):
    # Сортируем признаки и соответствующие им метки классов
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]
    
    # Находим пороги, которые делят отсортированные признаки на две подвыборки
    thresholds = (sorted_features[:-1] + sorted_features[1:]) / 2
    
    # Вычисляем доли объектов класса 1 и 0
    class_1_counts = np.cumsum(sorted_targets)
    class_0_counts = np.sum(sorted_targets) - class_1_counts
    
    # Вычисляем H(R) для всей выборки
    total_samples = len(sorted_targets)
    total_gini = 1 - (class_1_counts ** 2 / total_samples ** 2) - (class_0_counts ** 2 / total_samples ** 2)
    
    # Вычисляем критерий Джини для каждого порога
    left_sizes = np.arange(1, total_samples)
    right_sizes = total_samples - left_sizes
    left_ginis = 1 - ((class_1_counts[left_sizes] ** 2) / (left_sizes ** 2)) - ((class_0_counts[left_sizes] ** 2) / (left_sizes ** 2))
    right_ginis = 1 - ((class_1_counts[left_sizes] ** 2) / (right_sizes ** 2)) - ((class_0_counts[left_sizes] ** 2) / (right_sizes ** 2))
    
    # Вычисляем критерий Джини для каждого порога
    ginis = - (left_sizes / total_samples) * left_ginis - (right_sizes / total_samples) * right_ginis
    
    # Находим оптимальный порог и соответствующее значение критерия Джини
    best_split_index = np.argmin(ginis)
    threshold_best = thresholds[best_split_index]
    gini_best = ginis[best_split_index]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator):
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        if len(sub_y) < self._min_samples_split:
        # Условия останова: маленькая подвыборка, делаем узел терминальным
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if np.all(sub_y == sub_y[0]):
        # Все метки классов одинаковы, делаем узел терминальным
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_count / (current_click + 1)  # Добавляем 1, чтобы избежать деления на ноль
                sorted_categories = sorted(ratio, key=ratio.get, reverse=True)
                categories_map = {category: idx for idx, category in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError("Unknown feature type")

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                threshold_best = threshold
                split = feature_vector < threshold

        if gini_best is None:
            # Не удалось найти подходящий признак для разделения, делаем узел терминальным
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
        else:
            # Найден лучший сплит
            node["type"] = "nonterminal"
            node["feature_split"] = feature_best
            node["threshold"] = threshold_best
            node["left_child"], node["right_child"] = {}, {}
            self._fit_node(sub_X[split], sub_y[split], node["left_child"])
            self._fit_node(sub_X[~split], sub_y[~split], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            # Если признак числовой, проверяем порог и переходим в соответствующее поддерево
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature] == "categorical":
            # Если признак категориальный, проверяем, принадлежит ли значение категории
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError("Unknown feature type")


    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
