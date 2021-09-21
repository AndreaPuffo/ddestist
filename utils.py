import numpy as np
import numpy.linalg as la
import scipy as sc
import pandas as pd
import timeit
import theano
import torch
import theano.tensor as T
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from estimation import Estim_Net


from os import path


def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10**precision), 10**precision)


def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def find_paths(start, neighbors, n):
    if n == 0:
        return [[start]]

    paths = [[start] + path for next_state in neighbors[start][0,:]
             for path in find_paths(next_state, neighbors, n-1)]  # if start not in path]
    return paths


def find_ist_paths(start, neighbors, ists, n):
    if n == 0:
        return ists[start], 1
    paths = 0.
    n_p_tot = 0
    for neighbor in neighbors[start][0,:]:
        path, n_p = find_ist_paths(neighbor, neighbors, ists, n-1)
        # if start not in path:
        paths += path + ists[start]*n_p
        n_p_tot += n_p

    return paths, n_p_tot
    # paths = np.array([ists[start] + path for next_state in neighbors[start][0,:]
    #                   for path in find_ist_paths(next_state, neighbors, ists, n-1)])  # if start not in path]


def find_paths_exclude(start, neighbors, n, excludeSet = None):
    if excludeSet == None:
        excludeSet = set([start])
    else:
        excludeSet.add(start)
    if n == 0:
        return [[start]]
    paths = [[start]+path for neighbor in neighbors[start][0,:]
             if neighbor not in excludeSet
             for path in find_paths_exclude(neighbor, neighbors, n-1, excludeSet)]
    excludeSet.remove(start)
    return paths


def normalize_until(matr):
    loops = 0
    error_flag = True
    normalised_T = normalize(matr, norm='l1', axis=1)
    # normalised_T = np.around(normalised_T, decimals=6)
    while ((sc.sum(normalised_T, axis=1) == 1.).all() == False or loops<1e6) and error_flag:

        new_data = np.around(normalised_T.data, decimals=5)
        new_normal = sc.sparse.dok_matrix(matr.shape)
        for k, v in zip(matr.keys(), new_data):
            new_normal[k] = v
        normalised_T = normalize(new_normal, norm='l1', axis=1)
        loops += 1
        err_min = 1. - sc.sum(normalised_T, axis=1).min()
        err_max = - 1. + sc.sum(normalised_T, axis=1).max()
        # the only way to get out of the precision errors
        if abs(err_min) < 1e-10 and abs(err_max) < 1e-10:
            error_flag = False

    return normalised_T


def create_sampler_from_net(approx, net, X_train):
    # create symbolic input
    x = T.matrix("X")
    # symbolic number of samples is supported, we build vectorized posterior on the fly
    n = T.iscalar("n")
    # Do not forget test_values or set theano.config.compute_test_value = 'off'
    x.tag.test_value = np.empty_like(X_train[:100])
    n.tag.test_value = 1000
    _sample_proba = approx.sample_node(
        net.out.distribution.p, size=n, more_replacements={net["ann_input"]: x}
    )
    # It is time to compile the function
    # No updates are needed for Approximation random generator
    # Efficient vectorized form of sampling is used
    sample_proba = theano.function([x, n], _sample_proba)

    return sample_proba


def create_partitions(n_out, probab_ist, X, min_parts):

    predicted_ist = torch.argmax(probab_ist, dim=1).type(torch.FloatTensor)
    certain_features = [[] for i in range(n_out)]
    idx_features = [[] for i in range(n_out)]
    # BUILD SUPER PARTITIONS
    # uncertain
    unceert = []
    uncert_avg, uncert_max, uncert_min = 0., -1, n_out + 1
    prob_pct = min_parts * torch.ones(1, X.shape[0])
    for sample in range(probab_ist.shape[0]):
        idx_max_ = torch.sort(probab_ist[sample, :], descending=True).indices[0]
        idx_second_max_ = torch.sort(probab_ist[sample, :], descending=True).indices[1]
        # uncertainty
        if torch.abs(probab_ist[sample, idx_max_] - probab_ist[sample, idx_second_max_]) < 0.1:  # 1. / (n_out + 1):
            unceert.append(X[sample])
            # if uncertain, set the partition to n_out + 1 (which is n_out starting from zero)
            predicted_ist[sample] = n_out
            most_prob_ist = idx_max_
            if most_prob_ist > uncert_max:
                uncert_max = most_prob_ist.item()
            if most_prob_ist < uncert_min:
                uncert_min = most_prob_ist.item()
            uncert_avg += most_prob_ist
        else:
            certain_features[idx_max_] += [X[sample]]
            idx_features[idx_max_] += [sample]
            # saturate at 9 to avoid probability 1 reaching 10 prob_pct and "overflow"
            prob_pct[0, sample] = min(9, ((probab_ist[sample, idx_max_] * 100) // 10).item())

    return unceert, uncert_avg, uncert_max, uncert_min, \
           certain_features, idx_features, predicted_ist, prob_pct


def compute_trans_matrix_unctn_state(n_output, out_traces):
    # dim = n_output + 1 (uncertain)
    transition_matrix = np.zeros((n_output + 1, n_output + 1))
    for out_tr in out_traces:
        for idx in range(out_tr.shape[0] - 1):
            transition_matrix[int(out_tr[idx].item()), int(out_tr[idx + 1].item())] += 1
    norms = np.sum(transition_matrix, axis=1)
    # fix zeros in norms, replace with 1
    idxs_zeros = np.where(norms == 0)
    norms[idxs_zeros] = 1.

    normalised_T = (transition_matrix.T / norms).T
    normalised_T[idxs_zeros, idxs_zeros] = 1.
    return normalised_T


def compute_min_max_aist_unctn_state(aist_series, out_traces, selected_trace,
                                    uncert_values, len_sequences, normalised_T,
                                    n_output, X, predicted_ist, label=None):
    print('-' * 80)
    print(f'(True) ' + label + f' AIST: {aist_series[selected_trace]}')
    print(f'(Estim) ' + label + f' AIST: {out_traces[selected_trace].mean()}')

    uncert_min, uncert_avg, uncert_max = uncert_values

    corresponding_x_idx = sum(len_sequences[:selected_trace])
    first_x0 = X[corresponding_x_idx, :]
    first_ist = int(predicted_ist[corresponding_x_idx].item())

    pi = np.zeros((1, n_output + 1))
    pi[0, first_ist] = 1.

    aist_trsys = 0.
    min_aist_trsys = 0.
    max_aist_trsys = 0.
    avg_aist_trsys = 0.
    for i in range(len_sequences[selected_trace]):
        # compute aist_bounds
        aist_trsys = np.sum([i * pi[0, i] for i in range(pi.shape[1] - 1)])
        min_aist_trsys += aist_trsys + uncert_min * pi[0, -1]
        max_aist_trsys += aist_trsys + uncert_max * pi[0, -1]
        avg_aist_trsys += aist_trsys + uncert_avg * pi[0, -1]
        pi = pi @ normalised_T
    # print(f'Computed pi: {pi}')

    min_aist_trsys = min_aist_trsys / len_sequences[selected_trace]
    max_aist_trsys = max_aist_trsys / len_sequences[selected_trace]
    avg_aist_trsys = avg_aist_trsys / len_sequences[selected_trace]

    print(f'mAIST from transys: {min_aist_trsys}')
    print(f'MAIST from transys: {max_aist_trsys}')
    print(f'~AIST from transys: {avg_aist_trsys}')


def collapse_subparts_trmatrix(T, n_output, subparts):
    """
    from (n_out*subparts + 1) * (n_out*subparts + 1) matrix to a
    (n_out+1) * (n_out+1) matrix
    :param T:
    :param n_output:
    :param subparts:
    :return:
    """
    collapsed_t = np.zeros((n_output+1, n_output+1))
    # the partitions
    for i in range(n_output+1):
        for j in range(n_output+1):
            collapsed_t[i, j] = np.sum(T[i*subparts:(i+1)*subparts, j*subparts:(j+1)*subparts])  # /T[i*subparts:(i+1)*subparts, j*subparts:(j+1)*subparts].shape[0]
    # compute norms
    norms = np.sum(collapsed_t, axis=1)
    # fix zeros in norms, replace with 1
    idxs_zeros = np.where(norms == 0)
    norms[idxs_zeros] = 1.
    collapsed_t = (collapsed_t.T/norms).T
    collapsed_t[idxs_zeros, idxs_zeros] = 1.

    return collapsed_t


def compute_min_max_aist_with_pct(aist_series, out_traces, selected_trace,
                                  subparts, min_parts,
                                  uncert_values, len_sequences, normalised_T,
                                  n_output, X, predicted_ist, prob_pct, label=None):
    print('-' * 80)
    print(f'(True)' + label + f' AIST: {aist_series[selected_trace]}')
    print(f'(Estim)' + label + f' AIST: {out_traces[selected_trace].mean()}')

    uncert_min, uncert_avg, uncert_max = uncert_values

    corresponding_x_idx = sum(len_sequences[:selected_trace])
    first_x0 = X[corresponding_x_idx, :]
    first_ist = int(predicted_ist[corresponding_x_idx].item())
    first_pct = int(prob_pct[0, corresponding_x_idx].item()) - min_parts

    pi = np.zeros((1, n_output * subparts + 1))
    pi[0, first_ist * subparts + first_pct] = 1.

    aist_trsys = 0.
    min_aist_trsys = 0.
    max_aist_trsys = 0.
    avg_aist_trsys = 0.
    for i in range(len_sequences[selected_trace]):
        # compute aist_bounds
        aist_trsys = np.sum([(i // subparts) * pi[0, i] for i in range(pi.shape[1] - 1)])
        min_aist_trsys += aist_trsys + uncert_min * pi[0, -1]
        max_aist_trsys += aist_trsys + uncert_max * pi[0, -1]
        avg_aist_trsys += aist_trsys + uncert_avg * pi[0, -1]
        pi = pi @ normalised_T
    # print(f'Computed pi: {pi}')

    min_aist_trsys = min_aist_trsys / len_sequences[selected_trace]
    max_aist_trsys = max_aist_trsys / len_sequences[selected_trace]
    avg_aist_trsys = avg_aist_trsys / len_sequences[selected_trace]

    print(f'mAIST from transys: {min_aist_trsys}')
    print(f'~AIST from transys: {avg_aist_trsys}')
    print(f'MAIST from transys: {max_aist_trsys}')


def domain_splitting(lower, upper, n_partitions):
    """
    splits the domain into a smaller grid
    :param lower: array-like, lower bounds
    :param upper: array-like, upper bounds
    :param n_partitions: int
    """
    n = len(lower)
    # get new lower bounds for every dim
    split_lower = [np.linspace(lower[dim], upper[dim], n_partitions+1)[:-1]
                   for dim in range(n)]
    # combinations of them creates the new lower-left corner of each new partition
    new_lowers = np.array(list(itertools.product(*split_lower)))
    # for every dim, compute the size of the side
    dim_sizes = (upper - lower)/n_partitions
    # for every new lower-left corner, add dim_sizes to get upper-right corner
    new_uppers = new_lowers + dim_sizes
    # centers
    centers = new_lowers + dim_sizes/2

    return new_lowers, new_uppers, centers


def compute_ais_from_ss_parts(aist_series, out_traces, parts_samples, n_parts, n_input,
                              selected_trace, len_sequences, predicted_ist_ctr, trans_matrix,
                              label=''):

    corresponding_x_idx = sum(len_sequences[:selected_trace])
    print('-' * 80)
    print(f'(True) ' + label + f' AIST: {aist_series[selected_trace]}')
    print(f'(Estim) ' + label + f' AIST: {out_traces[selected_trace].mean()}')
    # first_x0 = X_tensor[selected_trace, :]
    first_part = parts_samples[corresponding_x_idx]

    pi = np.zeros((1, n_parts ** n_input))
    pi[0, first_part] = 1.

    aist_trsys = 0.
    # min_aist_trsys = 0.
    # max_aist_trsys = 0.
    # avg_aist_trsys = 0.
    for i in range(len_sequences[selected_trace]):
        # compute aist_bounds
        aist_trsys += (pi @ predicted_ist_ctr.numpy())[0]
        # min_aist_trsys += aist_trsys + uncert_min * pi[0, -1]
        # max_aist_trsys += aist_trsys + uncert_max * pi[0, -1]
        # avg_aist_trsys += aist_trsys + uncert_avg * pi[0, -1]
        pi = pi @ trans_matrix
    # print(f'Computed pi: {pi}')

    aist_trsys = aist_trsys / len_sequences[selected_trace]
    # min_aist_trsys = min_aist_trsys/len_sequences[selected_trace]
    # max_aist_trsys = max_aist_trsys/len_sequences[selected_trace]
    # avg_aist_trsys = avg_aist_trsys/len_sequences[selected_trace]

    # print(f'mAIST from transys: {min_aist_trsys}')
    # print(f'MAIST from transys: {max_aist_trsys}')
    print(f'~AIST from transys: {aist_trsys}')


def compute_aist_from_ss_parts_bounds(aist_series, out_traces, parts_samples, n_parts, n_input,
                                      selected_trace, len_sequences, predicted_ists,
                                      trans_matrix, label=''):

    predicted_ist_lb, predicted_ist_ctr, predicted_ist_ub = predicted_ists

    corresponding_x_idx = sum(len_sequences[:selected_trace])
    print('-' * 80)
    print(f'(True) ' + label + f' AIST: {aist_series[selected_trace]}')
    print(f'(Estim) ' + label + f' AIST: {out_traces[selected_trace].mean()}, '
                                f'Error {abs(aist_series[selected_trace] - out_traces[selected_trace].mean())}')
    # first_x0 = X_tensor[selected_trace, :]
    first_part = parts_samples[corresponding_x_idx]

    pi = np.zeros((1, n_parts ** n_input))
    pi[0, first_part] = 1.

    aist_ctr = 0.
    aist_lb = 0.
    aist_ub = 0.

    for i in range(len_sequences[selected_trace]):
        # compute aist_bounds
        aist_ctr += (pi @ predicted_ist_ctr.numpy())[0]
        aist_lb += (pi @ predicted_ist_lb.numpy())[0]
        aist_ub += (pi @ predicted_ist_ub.numpy())[0]

        pi = pi @ trans_matrix
    # print(f'Computed pi: {pi}')

    aist_ctr = aist_ctr / len_sequences[selected_trace]
    aist_lb = aist_lb / len_sequences[selected_trace]
    aist_ub = aist_ub / len_sequences[selected_trace]

    print(f'mAIST from transys: {aist_lb}')
    print(f'~AIST from transys: {aist_ctr}')
    print(f'MAIST from transys: {aist_ub}')


def compute_aist_from_adj_matrix(aist_series, out_traces, parts_samples, tot_partitions,
                                 selected_trace, len_sequences, predicted_ists, trans_matrix,
                                 label=''):

    predicted_ist_min, predicted_ist_ctr, predicted_ist_max = predicted_ists

    corresponding_x_idx = sum(len_sequences[:selected_trace])
    print('-' * 80)
    print(f'(True) ' + label + f' AIST: {aist_series[selected_trace]}')
    print(f'(Estim) ' + label + f' AIST: {out_traces[selected_trace].mean()}')
    # first_x0 = X_tensor[selected_trace, :]
    first_part = parts_samples[corresponding_x_idx]

    pi = sc.sparse.dok_matrix((1, tot_partitions))  # np.zeros((1, tot_partitions), dtype=np.float64)
    pi[0, first_part] = 1.

    start_compute_aist = timeit.default_timer()

    aist_trsys = 0.
    m_aist_trsys = 0.
    max_aist_trsys = 0.
    # avg_aist_trsys = 0.
    for i in range(len_sequences[selected_trace]):
        # compute aist_bounds
        aist_trsys += (pi @ predicted_ist_ctr.numpy())[0]
        m_aist_trsys += (pi @ predicted_ist_min.numpy())[0]
        max_aist_trsys += (pi @ predicted_ist_max.numpy())[0]

        # min_aist_trsys += aist_trsys + uncert_min * pi[0, -1]
        # max_aist_trsys += aist_trsys + uncert_max * pi[0, -1]
        # avg_aist_trsys += aist_trsys + uncert_avg * pi[0, -1]
        pi = pi @ trans_matrix
        pi = pi / np.sum(pi)
    # print(f'Computed pi: {pi}')

    aist_trsys = aist_trsys / len_sequences[selected_trace]
    min_aist_trsys = m_aist_trsys/len_sequences[selected_trace]
    max_aist_trsys = max_aist_trsys/len_sequences[selected_trace]
    # avg_aist_trsys = avg_aist_trsys/len_sequences[selected_trace]

    end_compute_aist = timeit.default_timer()

    print('-' * 80)
    print(f'mAIST from transys: {min_aist_trsys}')
    print(f'~AIST from transys: {aist_trsys}')
    print(f'MAIST from transys: {max_aist_trsys}')
    print(f'AIST(s) Computed. Total Elapsed Time: {end_compute_aist - start_compute_aist}')

    # pi = np.zeros((1, tot_partitions), dtype=np.float64)
    # pi[0, first_part] = 1.

    return aist_series[selected_trace], out_traces[selected_trace].mean(), \
           min_aist_trsys, aist_trsys, max_aist_trsys, \
           end_compute_aist-start_compute_aist


def compute_aist_x_ist_lumped(aist_series, out_traces, parts_samples, n_states,
                              selected_trace, len_sequences, predicted_ists, col_neig_indexes,
                              trans_matrix, label=''):

    predicted_ist_ctr = predicted_ists

    corresponding_x_idx = sum(len_sequences[:selected_trace])
    print('-' * 80)
    print(f'(True) ' + label + f' AIST: {aist_series[selected_trace]}')
    print(f'(Estim) ' + label + f' AIST: {out_traces[selected_trace].mean()}, '
                                f'Error {abs(aist_series[selected_trace] - out_traces[selected_trace].mean())}')
    # first_x0 = X_tensor[selected_trace, :]
    first_part = np.where([parts_samples[corresponding_x_idx] in sublist for sublist in col_neig_indexes])

    pi = np.zeros((1, n_states))
    pi[0, first_part] = 1.

    aist_ctr = 0.
    aist_lb = 0.
    aist_ub = 0.

    for i in range(len_sequences[selected_trace]):
        # compute aist_bounds
        aist_ctr += (pi @ predicted_ist_ctr.numpy())[0]
        # aist_lb += (pi @ predicted_ist_lb.numpy())[0]
        # aist_ub += (pi @ predicted_ist_ub.numpy())[0]

        pi = pi @ trans_matrix
    # print(f'Computed pi: {pi}')

    aist_ctr = aist_ctr / len_sequences[selected_trace]
    # aist_lb = aist_lb / len_sequences[selected_trace]
    # aist_ub = aist_ub / len_sequences[selected_trace]

    # print(f'mAIST from transys: {aist_lb}')
    print(f'~AIST from transys: {aist_ctr}')
    # print(f'MAIST from transys: {aist_ub}')


def find_contraction_const(samples):
    """

    :param samples:
    :return:
    """
    max_k = -1
    mean_k = torch.zeros(1, len(samples))
    # assume norm is in the first column
    for idx, trace in enumerate(samples):
        if len(trace) > 0:
            masked_trace = trace[trace > 1e-3]
            lip = masked_trace[1:] / masked_trace[:-1]
            k = torch.max(lip)
            mean_k[0, idx] = lip.mean()
            if k > max_k:
                max_k = k

    return max_k, mean_k.mean()


def load_samples(tmax, N_traces, sigma):
    """
    loads previously computed sampled - traces
    :param tmax:
    :param N_traces:
    :param sigma:
    :return:
    """

    sigma_str = ''.join(str(sigma).split('.'))

    basepath = path.dirname(__file__)

    folder_name = 'traces_lin/'
    name = 'tmax' + str(tmax) + '_t' + str(N_traces) + '_s' + sigma_str
    name_data = folder_name + name + '.pkl'
    name_lens = folder_name + name + '_len' + '.pkl'
    # filepath_data = path.abspath(path.join(basepath, "..", name_data))
    # filepath_lens = path.abspath(path.join(basepath, "..", name_lens))

    filepath_data = path.abspath(path.join(basepath, name_data))
    filepath_lens = path.abspath(path.join(basepath, name_lens))

    df = pd.read_pickle(filepath_data)
    X = df[['norm_xi', 'angles']].to_numpy()
    Y = df['ist'].to_numpy(dtype=int)

    df_lens = pd.read_pickle(filepath_lens)
    len_sequences = df_lens['len_sequences'].to_numpy()

    return X, Y, len_sequences


def data_split_and_prep(X, Y, len_sequences, n_output):
    """

    :param X:
    :param Y:
    :param len_sequences:
    :param n_output:
    :return:
    """
    Y_outputs = np.zeros((X.shape[0], n_output))
    for idx in range(X.shape[0]):
        Y_outputs[idx, Y[idx]] = 1.0

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_outputs, test_size=0.6, shuffle=False)
    X_tensor = torch.tensor(X_train).type(torch.FloatTensor)
    Y_tensor = torch.tensor(Y_train).type(torch.FloatTensor)
    X_cv_tensor = torch.tensor(X_test).type(torch.FloatTensor)
    Y_cv_tensor = torch.tensor(Y_test).type(torch.FloatTensor)
    # test_splits randomises the data, reconstruct Y
    allX = torch.vstack([X_tensor, X_cv_tensor]).type(torch.FloatTensor)
    allY = torch.vstack([Y_tensor, Y_cv_tensor])
    all_ist = torch.argmax(allY, dim=1).type(torch.FloatTensor)

    aist_series, ist_series, traces = [], [], []
    idx_so_far = 0
    for l in range(len(len_sequences)):
        traces += [allX[idx_so_far:idx_so_far + len_sequences[l], :]]
        aist_series += [all_ist[idx_so_far:idx_so_far + len_sequences[l]].mean()]
        ist_series += [all_ist[idx_so_far:idx_so_far + len_sequences[l]]]
        idx_so_far += len_sequences[l]

    all_norms = [tr[:, 0] for tr in traces]

    max_k, mean_k = find_contraction_const(all_norms)

    print('AIST (train data): \n', aist_series)
    print(f'Mean AIST (train data): {np.array(aist_series).mean()}')

    return X_tensor, Y_tensor, X_cv_tensor, Y_cv_tensor, allX, aist_series


def learning_net(n_input, n_hidden_neurons, n_output, n_epochs, X_tr, Y_tr, X_cv, Y_cv):

    print('Classical Learning....')
    start_lrn = timeit.default_timer()
    nnet1 = Estim_Net(n_input, n_hidden_neurons, n_output,
                      n_epochs=int(n_epochs))
    nnet1.learn(x_train=X_tr, y=Y_tr, batch_size=50)
    end_lrn = timeit.default_timer()
    print(f'Done. Elapsed Time: {end_lrn - start_lrn} sec')
    yhat, _ = nnet1.forward(X_tr)
    mse_train = torch.norm(yhat - Y_tr) / Y_tr.shape[0]
    print(f'MSE: {mse_train}')
    spot_on_predictions = sum(torch.argmax(Y_tr, dim=1) - torch.argmax(yhat, dim=1) == 0)
    accuracy_train = spot_on_predictions / Y_tr.shape[0] * 100
    print(f'Accuracy: {accuracy_train} %')

    yhat_cv, _ = nnet1.forward(X_cv)
    mse_cv = torch.norm(yhat_cv - Y_cv) / Y_cv.shape[0]
    print(f'MSE (CV): {mse_cv}')
    spot_on_predictions = sum(torch.argmax(Y_cv, dim=1) - torch.argmax(yhat_cv, dim=1) == 0)
    accuracy_cv = spot_on_predictions / Y_cv.shape[0] * 100
    print(f'Accuracy (CV): {accuracy_cv} %')

    return nnet1, mse_train, accuracy_train, mse_cv, accuracy_cv, end_lrn-start_lrn


def build_ss_abstraction(n_parts, X, nnet, allX, len_sequences):
    """

    :param n_parts:
    :param X:
    :param nnet:
    :param allX:
    :param len_sequences:
    :return:
    """


    lb = my_floor(np.min(X, axis=0), precision=1)
    ub = my_ceil(np.max(X, axis=0), precision=1) + 0.1

    start_build_abstraction = timeit.default_timer()

    parts_lb, parts_ub, parts_ctr = domain_splitting(lb, ub, n_parts)
    # compute IST for each partition (assuming the center is informative)
    probab_ist_ctr, _ = nnet.forward(torch.tensor(parts_ctr).type(torch.FloatTensor))
    probab_ist_traces, _ = nnet.forward(allX)
    predicted_ist_ctr = torch.argmax(probab_ist_ctr, dim=1).type(torch.FloatTensor)
    predicted_ist_traces = torch.argmax(probab_ist_traces, dim=1).type(torch.FloatTensor)

    # lower and upper bound IST computation
    probab_ist_lb, _ = nnet.forward(torch.tensor(parts_lb).type(torch.FloatTensor))
    probab_ist_ub, _ = nnet.forward(torch.tensor(parts_ub).type(torch.FloatTensor))
    predicted_ist_lb = torch.argmax(probab_ist_lb, dim=1).type(torch.FloatTensor)
    predicted_ist_ub = torch.argmax(probab_ist_ub, dim=1).type(torch.FloatTensor)

    predicted_ist_min = torch.min(torch.stack([predicted_ist_lb, predicted_ist_ctr, predicted_ist_ub]), dim=0).values
    predicted_ist_max = torch.max(torch.stack([predicted_ist_lb, predicted_ist_ctr, predicted_ist_ub]), dim=0).values

    def find_partitions(samples):
        prods = [n_parts ** idx for idx in reversed(range(samples.shape[1]))]
        parts_number = np.sum(np.floor((samples - lb) / (ub - lb) * n_parts) * prods, axis=1)
        return (parts_number).astype(int)

    # partitions_ofthe_centers = find_partitions(parts_ctr)
    # for idx_c, c in enumerate(partitions_ofthe_centers):
    #     ax.annotate(c, (parts_ctr[idx_c, 0], parts_ctr[idx_c, 1]))

    # TRANSITION MATRIX
    parts_samples = find_partitions(X)

    # divide data into traces
    out_traces, x_traces = [], []
    idx_so_far = 0
    for l in range(len(len_sequences)):
        out_traces += [predicted_ist_traces[idx_so_far:idx_so_far + len_sequences[l]]]
        x_traces += [X[idx_so_far:idx_so_far + len_sequences[l]]]
        idx_so_far += len_sequences[l]

    # dim = n_output + 1 (uncertain)
    n_input = nnet.fc1.in_features
    overhead_trace = 0
    tot_partitions = n_parts ** n_input
    # instatiate sparse matrix
    adj_matrix = sc.sparse.dok_matrix((tot_partitions, tot_partitions))  # , dtype=np.float32)
    for x_tr in x_traces:
        for idx in range(x_tr.shape[0] - 1):
            start = parts_samples[idx + overhead_trace]
            targt = parts_samples[idx + overhead_trace + 1]
            adj_matrix[start, targt] = 1
        overhead_trace += x_tr.shape[0]
    norms = sc.sum(adj_matrix, axis=1)
    # fix zeros in norms, replace with 1
    idxs_zeros = np.where(norms == 0)
    norms[idxs_zeros] = 1.
    adj_matrix[idxs_zeros, idxs_zeros] = 1.

    end_build_abstraction = timeit.default_timer()
    print('-' * 80)
    print(f'Abstraction Built. Number of states: {tot_partitions}.')
    print(f'Elapsed Time: {end_build_abstraction - start_build_abstraction}')

    print(f'Number of non-deterministic states: {sc.sum(norms > 1.)}')

    # plots
    color_plots = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gold', 'grey',
                   'seagreen', 'deeppink', 'maroon', 'salmon', 'teal', 'khaki', 'crimson']
    # plt.figure()
    # for ctr_idx, ctr in enumerate(parts_ctr):
    #     plt.scatter(ctr[0], ctr[1], label=ctr_idx)
    # plt.legend()
    # plt.grid()
    if n_parts < 101:
        fig, ax = plt.subplots()
        prdct_ist_np = predicted_ist_ctr.numpy()
        for g in np.unique(prdct_ist_np):
            ix = np.where(prdct_ist_np == g)
            ax.scatter(parts_ctr[ix][:, 0], parts_ctr[ix][:, 1], c=color_plots[int(g)],
                       label=g, s=3)
        plt.grid()
        plt.legend()

    return adj_matrix, predicted_ist_min, predicted_ist_ctr, predicted_ist_max, \
           out_traces, parts_samples, end_build_abstraction-start_build_abstraction


def save_results_pkl(tmax, N_traces, sigma, n_hidden_neurons,
                     mse_train, accuracy_train, mse_cv, accuracy_cv, lrn_time,
                     tot_partitions, abs_time, aist_performance_idx):
    """

    :param tmax:
    :param N_traces:
    :param sigma:
    :param n_hidden_neurons:
    :param mse_train:
    :param accuracy_train:
    :param mse_cv:
    :param accuracy_cv:
    :param lrn_time:
    :param tot_partitions:
    :param abs_time:
    :param aist_performance_idx:
    :return:
    """

    data_df = pd.DataFrame(data=np.atleast_2d([tmax, N_traces, sigma]), columns=['tmax', 'n_traces', 'sigma'])
    train_df = pd.DataFrame(data=np.atleast_2d([
        n_hidden_neurons, mse_train.detach().numpy(), accuracy_train, mse_cv.detach().numpy(), accuracy_cv, lrn_time
        ]),
        columns=['n_hidden', 'mse_tr', 'acc_tr', 'mse_cv', 'acc_cv', 'lrn_time'])
    abs_df = pd.DataFrame(data=np.atleast_2d([tot_partitions, abs_time]),
                          columns=['n_abs_state', 'abs_time'])
    aist_comp_df = pd.DataFrame(data=np.atleast_2d([
        aist_performance_idx
            ]),
        columns=[
            'true_aist_rnd', 'estim_aist_rnd', 'mAist_rnd', 'aAIST_rnd', 'MAIST_rnd', 'time_rnd',
            'true_aist_low', 'estim_aist_low', 'mAist_low', 'aAIST_low', 'MAIST_low', 'time_low',
            'true_aist_hig', 'estim_aist_hig', 'mAist_hig', 'aAIST_hig', 'MAIST_hig', 'time_hig'
        ])

    # stack the dfs
    final_df = pd.concat([pd.concat([data_df.T, train_df.T, abs_df.T, aist_comp_df.T])]).T
    folder_out = 'results/'
    sigma_str = ''.join(str(sigma).split('.'))
    name = 'tmax' + str(tmax) + '_t' + str(N_traces) + '_s' + sigma_str
    name_out = folder_out + name + '_' + str(timeit.default_timer()) + '.pkl'
    final_df.to_pickle(name_out)


