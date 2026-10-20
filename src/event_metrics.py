import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def metric_row(p, empty_recall=np.nan):
    y, alarm = p.positive.to_numpy(bool), p.score.to_numpy() > p.cutoff.to_numpy()
    tp, fp, fn, tn = [int(z.sum()) for z in [y & alarm, ~y & alarm, y & ~alarm, ~y & ~alarm]]
    starts = p.onset.to_numpy(bool)
    row = {'n': len(p), 'positives': int(y.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
           'precision': tp/(tp+fp) if tp+fp else 0., 'recall': tp/(tp+fn) if tp+fn else empty_recall,
           'F1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
           'AP': average_precision_score(y, p.score) if y.any() else np.nan,
           'ROC_AUC': roc_auc_score(y, p.score) if len(set(y)) == 2 else np.nan,
           'Brier': brier_score_loss(y, p.score) if p.probability.all() else np.nan,
           'onsets': int(starts.sum()), 'onsets_detected': int((starts & y & alarm).sum()),
           'onset_recall': float((starts & y & alarm).sum()/starts.sum()) if starts.any() else np.nan,
           'continuation_tp': int((y & alarm & ~starts).sum()),
           'false_alarms_per_12_observed_months': fp*365.2425/7/len(p)}
    if p.predicted.notna().all():
        error = abs(p.actual-p.predicted)
        denominator = abs(p.actual)+abs(p.predicted)
        row.update(RMSE=float(np.sqrt(np.mean(error**2))), MAE=float(error.mean()),
                   MASE=float((error/p.mase_scale).mean()),
                   sMAPE=float(np.mean(np.divide(200*error, denominator, out=np.zeros(len(p)), where=denominator.to_numpy()!=0))))
    return row
