# VENDORED VERBATIM from the authors' MDCP repo (Tawachi & Laufer-Goldshtein,
# ICLR 2025): branch di-metric, jackknife_standalone/yams_jacknife.py, git blob
# 7ed75b3 -- itself a byte-verbatim copy of R2CCP/yams_jacknife.py on branch
# paper-version (the authors' only verified-working Jackknife+ Multi-Score CP,
# paper Algorithm B.1). Everything below this header is UNMODIFIED; known
# quirks (votes over k' unique centers with floor threshold, argmax empty-set
# fallback, np.delete(vec, i) assuming sorted-unique center i == cal row i)
# are kept as-is for fidelity. Consumed by src/mdcp_jackknife_compare.py.
from scipy.spatial import KDTree
import numpy as np

#TODO- make this comatible with the weighted cells method

def create_true_rest_sets(scores: np.ndarray,target: np.ndarray,stage: str)-> (tuple,tuple):
    rest_preds=np.zeros([scores.shape[0],scores.shape[1],scores.shape[2]-1])
    true_preds=np.zeros([scores.shape[0],scores.shape[1]])
    if stage=='CAL':
        for head in range(scores.shape[0]):
            for row in range(scores.shape[1]):
                selected_elements=np.delete(scores[head,row,:], int(target[row]), axis=0)
                rest_preds[head,row,:]=selected_elements
            true_preds[head]=scores[head,:,:][np.arange(len(scores[head,:,:])), target.astype(int)]
        rest_preds_concat =rest_preds
    elif stage=='VAL':
        results = [(scores[:, :, cl], np.delete(scores, cl, axis=2)) for cl in range(scores.shape[2])]
        true_preds, rest_preds_concat = zip(*results)
    return true_preds, rest_preds_concat


def find_closest_columns(array1, array2):
    tree = KDTree(array1.T)
    self_closenes=tree.query(array1.T, k=2)[1][:,1]
    l_first, l_second=[],[]
    for r in range(array2.shape[1]):
        closest_indices = tree.query(array2[:,r,:].T, k=2)[1]
        first_closest,second_closest=closest_indices[:,0],closest_indices[:,1]
        l_first.append(first_closest),l_second.append(second_closest)
    return np.array(l_first),np.array(l_second),np.array(self_closenes)

def create_n_maps(first_closest_idxs,second_closest_idxs,count_cal_points_in_center,phase):
    n_centers=count_cal_points_in_center.shape[0]
    all_scores_cal=[]
    all_test_attribution=[]
    for i in range(n_centers):
        vec=first_closest_idxs.copy()
        vec[first_closest_idxs==i]=second_closest_idxs[first_closest_idxs==i]

        if phase=='cal':
            vec_i = np.delete(vec, i, axis=0).flatten()
            index_vec, count_vec = np.unique(vec_i,return_counts=True)
            total_score = np.ones(n_centers)
            total_score[i]=0
            total_score[index_vec] = +count_vec
            total_score = (total_score / count_cal_points_in_center)+1
            all_scores_cal.append(total_score)
        elif phase=='test':
            all_test_attribution.append(vec)

    return np.array(all_scores_cal) ,all_test_attribution


def create_wanted_bins_centers_part_1(cal_true_preds, cal_rest_preds,phase):
    centers_unique, count_cal_points_in_center = np.unique(cal_true_preds, axis=1, return_counts=True)
    first_closest_idxs,second_closest_idxs,self_closenes_idx = find_closest_columns(centers_unique,cal_rest_preds)
    total_score_mat_cal,all_test_attribution_l=create_n_maps(first_closest_idxs, second_closest_idxs, count_cal_points_in_center,phase)
    if phase=='cal':
        self_score = total_score_mat_cal[np.arange(total_score_mat_cal.shape[0]), self_closenes_idx]
        return self_score,total_score_mat_cal
    elif phase=='test':
        return all_test_attribution_l,None


def allocate_points(precal_scores, precal_target,a,b,test_scores,test_target,alpha,config):
    test_scores = fine_grid_test_score(test_scores, config["fine_grid_test_score"][1]) if \
    config["fine_grid_test_score"][0] else test_scores
    test_target = test_target * config["fine_grid_test_score"][1] if config["fine_grid_test_score"][0] else test_target
    cal_true_preds, cal_rest_preds = create_true_rest_sets(precal_scores, precal_target, 'CAL')
    self_centers_scores ,total_score_mat_cal= create_wanted_bins_centers_part_1(cal_true_preds, cal_rest_preds,phase='cal')
    all_test_attribution_l ,_= create_wanted_bins_centers_part_1(cal_true_preds, test_scores,phase='test')
    coverage, mean_set ,result_mat= test_bins(all_test_attribution_l, test_target, self_centers_scores,total_score_mat_cal, alpha)
    return coverage,mean_set,result_mat,(None,None),None

def test_bins(all_test_attribution_l, test_target, self_centers_scores,total_score_mat_cal, alpha):
    n_of_centers=len(all_test_attribution_l)
    map_shape=all_test_attribution_l[0].shape
    scores_tensor=[]
    for i in range(n_of_centers):
        cur_map=all_test_attribution_l[i].flatten()
        scores_tensor.append(total_score_mat_cal[i, cur_map].reshape(map_shape) >= self_centers_scores[i])
    scores_tensor=np.array(scores_tensor)
    mat_sum_results=scores_tensor.sum(axis=0)
    alpha_hat = int((n_of_centers + 1) * (1 - alpha))
    result_mat=mat_sum_results<=alpha_hat

    zero_rows=result_mat.sum(axis=1) == 0
    if zero_rows.any():
        force_one_idx=np.argmax(mat_sum_results[zero_rows,:],axis=1)
        result_mat[zero_rows,force_one_idx]=True

    return result_mat[range(len(result_mat)), test_target.astype(int)].sum() / len(result_mat), result_mat.sum(
        axis=1).mean(),result_mat

def fine_grid_test_score(test_score,grid_res) :
    B, S, P = test_score.shape
    x_old = np.arange(P)
    x_new = np.linspace(0, P - 1, P * grid_res)
    out = np.empty((B, S, len(x_new)), dtype=test_score.dtype)
    for b in range(B):
        for s in range(S):
            out[b, s] = np.interp(x_new, x_old, test_score[b, s])
    return out
