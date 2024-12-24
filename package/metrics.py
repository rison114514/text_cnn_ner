from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def calculate_metrics(true_labels, pred_labels, label_map):
    """
    计算NER任务的评估指标：精确率、召回率、F1分数和准确率

    Args:
        true_labels (list of int): 真实标签
        pred_labels (list of int): 预测标签
        label_map (dict): 从标签到索引的映射

    Returns:
        dict: 包含精确率、召回率、F1分数和准确率的字典
    """
    # 打印调试信息
    print(f"Calculating metrics for {len(true_labels)} true labels and {len(pred_labels)} predicted labels")

    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0,
                                labels=list(label_map.values()))
    recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0, labels=list(label_map.values()))
    f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0, labels=list(label_map.values()))
    accuracy = accuracy_score(true_labels, pred_labels)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def calculate_entity_level_metrics(true_labels, pred_labels, label_map, batch_size):
    """
    计算实体级别的评估指标

    Args:
        true_labels (list of int): 真实标签
        pred_labels (list of int): 预测标签
        label_map (dict): 从标签到索引的映射
        batch_size (int): 批次大小

    Returns:
        dict: 包含精确率、召回率、F1分数的字典
    """

    def extract_entities(labels):
        entities = []
        entity = []
        for idx, label in enumerate(labels):
            if label.startswith('B-'):
                if entity:
                    entities.append(entity)
                    entity = []
                entity = [label[2:], idx]
            elif label.startswith('I-') and entity:
                if label[2:] == entity[0]:
                    entity.append(idx)
                else:
                    entities.append(entity)
                    entity = []
            else:
                if entity:
                    entities.append(entity)
                    entity = []
        if entity:
            entities.append(entity)
        return entities

    # 将展平的标签列表转换回嵌套列表
    seq_length = len(true_labels) // batch_size  # 使用传递的批次大小
    true_labels_nested = [true_labels[i:i + seq_length] for i in range(0, len(true_labels), seq_length)]
    pred_labels_nested = [pred_labels[i:i + seq_length] for i in range(0, len(pred_labels), seq_length)]

    true_entities = [extract_entities([list(label_map.keys())[label] for label in seq]) for seq in true_labels_nested]
    pred_entities = [extract_entities([list(label_map.keys())[label] for label in seq]) for seq in pred_labels_nested]

    true_entity_set = set(tuple(entity) for entities in true_entities for entity in entities)
    pred_entity_set = set(tuple(entity) for entities in pred_entities for entity in entities)

    correct_entities = true_entity_set & pred_entity_set
    precision = len(correct_entities) / len(pred_entity_set) if pred_entity_set else 0
    recall = len(correct_entities) / len(true_entity_set) if true_entity_set else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
