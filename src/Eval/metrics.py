"""Classification metrics with no optional statistical dependencies."""


def classification_metrics(targets, predictions, class_names):
    if len(targets) != len(predictions):
        raise ValueError("Targets and predictions must have the same length")
    if not targets:
        raise ValueError("Cannot score an empty split")
    count = len(class_names)
    confusion = [[0] * count for _ in range(count)]
    for target, prediction in zip(targets, predictions):
        if not (0 <= target < count and 0 <= prediction < count):
            raise ValueError("Class index is outside the declared taxonomy")
        confusion[target][prediction] += 1
    per_class = []
    for index, name in enumerate(class_names):
        correct = confusion[index][index]
        support = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        precision = correct / predicted if predicted else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class_name": name, "precision": precision,
                          "recall": recall, "f1": f1, "support": support})
    return {"accuracy": sum(confusion[i][i] for i in range(count)) / len(targets),
            "macro_f1": sum(row["f1"] for row in per_class) / count,
            "samples": len(targets), "per_class": per_class,
            "class_names": list(class_names), "confusion_matrix": confusion,
            "confusion_matrix_axes": {"rows": "true", "columns": "predicted"}}
