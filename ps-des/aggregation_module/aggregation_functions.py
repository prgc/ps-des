from deslib.util import aggregation



def aggregation_predictions(y_pred, y_proba, str_aggregation):
    classe = get_aggregation(y_pred, y_proba, str_aggregation)
    return classe


def get_aggregation(y_pred, y_proba, str_aggregation):
    y_xq = []
    if (str_aggregation == 'majority'):
        y_xq = get_votes_aggregation(y_pred, y_proba, str_aggregation)
    else:
        y_xq = get_proba_aggregation(y_pred, y_proba, str_aggregation)
    return y_xq

            
def get_votes_aggregation(y_pred, y_proba, str_aggregation):
    y_xq = []
    for run in y_pred:
        y_xq_temp = []
        for votes in run:
            if (str_aggregation == 'majority'):
                classe = aggregation.majority_voting_rule([votes])
            y_xq_temp.append(int(classe[0]))
        y_xq.append(y_xq_temp)
    return y_xq

def get_proba_aggregation(y_pred, y_proba, str_aggregation):
    y_xq = []
    for run in y_proba:
        y_xq_temp = []
        for votes in run:
            if (str_aggregation == 'average'):
                classe = aggregation.average_rule([votes])
            if (str_aggregation == 'max'):
                classe = aggregation.maximum_rule([votes])
            if (str_aggregation == 'min'):
                classe = aggregation.minimum_rule([votes])
            if (str_aggregation == 'median'):
                classe = aggregation.median_rule([votes])
            if (str_aggregation == 'product'):
                classe = aggregation.product_rule([votes])            
            y_xq_temp.append(int(classe[0]))
        y_xq.append(y_xq_temp)
    return y_xq