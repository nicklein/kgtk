import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator


# Chooses values for the epsilon and min_samples parameters of DBSCAN to be used for clustering given labels.
def choose_DBSCAN_params(labels_df):
    values = np.array(labels_df.loc[:,"node2"]).reshape(-1, 1)
    
    # edge case only one sample will give an error if we try to compute knn.
    if len(values) < 2:
        print("only one sample for label:\n{}\n".format(labels_df))
        return (1, 1)
    
    # min_samples would ideally be set with some domain-insight.
    # To make this automated, we'll use a heuristic - ln(number of data points)
    min_samples = int(np.floor(np.log(len(values))))
    min_samples = max(min_samples, 1) # don't choose a number less than 1
    
    # epsilon can be chosen by plotting the distances of the k'th nearest neighbor from each point
    # where k is min_samples. Points that belong to clusters should have smaller distances, whereas
    # noise points can have distances that are much farther. We'll look for a knee in this graph to set epsilon.
    neigh = NearestNeighbors(n_neighbors = min_samples + 1) # +1 so we find k'th nearest neighbor not including the point iteslf.
    values_neigh = neigh.fit(values)
    distances, indices = values_neigh.kneighbors(values)
    distances = np.sort(distances[:,min_samples], axis = 0)
    
    kneedle = KneeLocator(range(len(distances)), distances, S=1, curve='convex', direction='increasing', interp_method = 'polynomial')
    epsilon = kneedle.knee_y
    if epsilon == None:
        print("no knee found for these labels:\n{}".format(labels_df))
        epsilon = distances[len(distances) // 2]
        print("using median distance to k nearest neighbor instead ({})\n".format(epsilon))
    
    return (min_samples, epsilon)


# Uses DBSCAN to assign values to clusters
def DBSCAN_1d_values(values, eps, min_samples):
    # dbscan doesn't like eps == 0. This translates to not doing any bucketing.
    # handling this as an edge case so dbscan doesn't throw an exception
    if eps <= 0:
        return values # same-valued points are labeled the same
    
    # dbscan expects multiple dimensions. Edit new object instead of existing view of df.
    values = np.array(values)
    values = np.append(values.reshape(-1,1), np.zeros((len(values),1)), axis = 1)
        
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(values)

    return (db.labels_)


# Given values that have been assigned to clusters,
# return a list of intervals that are consistent with these clusters
def get_intervals_for_values_based_on_clusters(values, labels):
    
    values = np.array(values)
    labels = np.array(labels)
    indexes = np.arange(len(values))
    
    # sort values and corresponding labels in ascending order
    labels = np.array([l for l, v in sorted(zip(labels, values), key=lambda pair: pair[1])])
    indexes = np.array([i for i, v in sorted(zip(indexes, values), key=lambda pair: pair[1])])
    values.sort()
    
    intervals = [(None, None)] # initially a single interval with no lower or upper bound
    
    # create intervals
    cur_label = labels[0]
    for i in range(len(labels)):
        # if new label, set upper bound of previous interval,
        # and start lower bound of a new interval.
        if labels[i] != cur_label:
            prev_interval_lb = (intervals[-1])[0]
            new_interval_edge = values[i-1] + ((values[i] - values[i-1]) / 2)
            intervals[-1] = (prev_interval_lb, new_interval_edge)
            intervals.append((new_interval_edge, None))
            cur_label = labels[i]
    
    # assign intervals to values
    cur_label = labels[0]
    cur_interval_ix = 0
    intervals_for_values = []
    for i in range(len(labels)):
        if labels[i] != cur_label:
            cur_interval_ix += 1
            cur_label = labels[i]
        intervals_for_values.append(intervals[cur_interval_ix])
    
    # rearrange intervals to original order of values
    intervals_for_values_unscrambled = np.zeros(len(intervals_for_values), dtype=tuple)
    for i in range(len(indexes)):
        intervals_for_values_unscrambled[indexes[i]] = intervals_for_values[i]
    
    return intervals_for_values_unscrambled

# Given a file containing numeric valued attribute labels,
# discretize these labels create a new file with the resulting attribute interval labels
def discretize_labels_DBSCAN(avl_file_in, ail_file_out):
    df = pd.read_csv(avl_file_in, delimiter='\t')
    
    # add lower bound and upper bound columns
    df.insert(loc = len(df.columns), column = "lower_bound", value = ["" for i in range(df.shape[0])])
    df.insert(loc = len(df.columns), column = "upper_bound", value = ["" for i in range(df.shape[0])])
    
    # blank values in units columns are expected as not all values will have units.
    # we want blank units to compare equal to eachother, so fill NaN's in as "" in units columns.
    if "si_units" in df.columns and "wd_units" in df.columns:
        df.fillna("", inplace = True)
        
    # we also don't want to consider any string type values
    types = [type(v) for v in df.loc[:,"node2"]]
    non_str_mask = [True if (t != str) else False for t in types]
    df = df.loc[non_str_mask]
    
    # get distinct label types (defined by type and property, as well as si and wd units if we have them)
    if "si_units" in df.columns and "wd_units" in df.columns:
        distinct_labels = df.loc[:, ["node1", "label", "si_units", "wd_units"]].drop_duplicates()
    else:
        distinct_labels = df.loc[:, ["node1", "label"]].drop_duplicates()
        
    # Could probably be improved with a list comprehension
    for index, row in distinct_labels.iterrows():
        # Get subset of labels that match this distinct kind of label
        subset_mask = (df["node1"] == row["node1"]) & (df["label"] == row["label"])
        # if we have units, treat these as part of the kind of label
        if "si_units" in df.columns and "wd_units" in df.columns:
            subset_mask = subset_mask & (df["si_units"] == row["si_units"]) & (df["wd_units"] == row["wd_units"])
        subset = df.loc[subset_mask]
        
        # Shouldn't happen, just checking.
        values = subset.loc[:,"node2"]
        if(len(values) == 0):
            print("no values found for subset:\n{}\n".format(subset))
            print("row:\n{}\n".format(row))
        
        min_samples, epsilon = choose_DBSCAN_params(subset)
        cluster_labels = DBSCAN_1d_values(values, epsilon, min_samples)
        intervals_for_values = get_intervals_for_values_based_on_clusters(values, cluster_labels)
        lbounds = [pair[0] for pair in intervals_for_values]
        ubounds = [pair[1] for pair in intervals_for_values]
        df.loc[subset_mask,"lower_bound"] = lbounds
        df.loc[subset_mask,"upper_bound"] = ubounds
    
    df.to_csv(ail_file_out, sep = '\t', index = False)
    
    
    
# Given a file containing numeric valued attribute labels,
# discretize these labels create a new file with the resulting attribute interval labels
def discretize_labels_fixed_width(avl_file_in, ail_file_out, width):
    df = pd.read_csv(avl_file_in, delimiter='\t')
    
    # add lower bound and upper bound columns
    df.insert(loc = len(df.columns), column = "lower_bound", value = ["" for i in range(df.shape[0])])
    df.insert(loc = len(df.columns), column = "upper_bound", value = ["" for i in range(df.shape[0])])
        
    # we don't want to consider any string type values
    types = [type(v) for v in df.loc[:,"node2"]]
    non_str_mask = [True if (t != str) else False for t in types]
    df = df.loc[non_str_mask]
    
    values = df.loc[:,"node2"]
    df.loc[:,"lower_bound"] = values - values.mod(width)
    df.loc[:,"upper_bound"] = values - values.mod(width) + width
    
    df.to_csv(ail_file_out, sep = '\t', index = False)
    

    
    
    
# given values that all correspond to the same label kind,
# return corresponding intervals based on percentile
def get_intervals_for_values_based_on_percentile(values, num_bins):
    values = np.array(values)
    indexes = np.arange(len(values))
    
    # sort values and corresponding labels in ascending order
    indexes = np.array([i for i, v in sorted(zip(indexes, values), key=lambda pair: pair[1])])
    values.sort()
    
    interval_bounds = []
    for i in range(num_bins):
        index_of_lbound = int(((i)/num_bins)*len(values))
        index_of_ubound = int(((i+1)/num_bins)*len(values)) - 1
        lbound = values[index_of_lbound]
        ubound = values[index_of_ubound]
        interval_bounds.append((lbound,ubound))
    intervals_for_values=[]
    cur_interval_idx=0
    for i in range(len(values)):
        while values[i] > interval_bounds[cur_interval_idx][1]:
            cur_interval_idx += 1
        intervals_for_values.append(interval_bounds[cur_interval_idx])
    
    # rearrange intervals to original order of values
    intervals_for_values_unscrambled = np.zeros(len(intervals_for_values), dtype=tuple)
    for i in range(len(indexes)):
        intervals_for_values_unscrambled[indexes[i]] = intervals_for_values[i]
        
    return intervals_for_values_unscrambled
    
    
# Given a file containing numeric valued attribute labels,
# discretize these labels create a new file with the resulting attribute interval labels
def discretize_labels_by_percentile(avl_file_in, ail_file_out, num_bins):
    df = pd.read_csv(avl_file_in, delimiter='\t')
    
    # add lower bound and upper bound columns
    df.insert(loc = len(df.columns), column = "lower_bound", value = ["" for i in range(df.shape[0])])
    df.insert(loc = len(df.columns), column = "upper_bound", value = ["" for i in range(df.shape[0])])
    
    # blank values in units columns are expected as not all values will have units.
    # we want blank units to compare equal to eachother, so fill NaN's in as "" in units columns.
    if "si_units" in df.columns and "wd_units" in df.columns:
        df.fillna("", inplace = True)
        
    # we also don't want to consider any string type values
    types = [type(v) for v in df.loc[:,"node2"]]
    non_str_mask = [True if (t != str) else False for t in types]
    df = df.loc[non_str_mask]
    
    # get distinct label types (defined by type and property, as well as si and wd units if we have them)
    if "si_units" in df.columns and "wd_units" in df.columns:
        distinct_labels = df.loc[:, ["node1", "label", "si_units", "wd_units"]].drop_duplicates()
    else:
        distinct_labels = df.loc[:, ["node1", "label"]].drop_duplicates()
        
    # Could probably be improved with a list comprehension
    for index, row in distinct_labels.iterrows():
        # Get subset of labels that match this distinct kind of label
        subset_mask = (df["node1"] == row["node1"]) & (df["label"] == row["label"])
        # if we have units, treat these as part of the kind of label
        if "si_units" in df.columns and "wd_units" in df.columns:
            subset_mask = subset_mask & (df["si_units"] == row["si_units"]) & (df["wd_units"] == row["wd_units"])
        subset = df.loc[subset_mask]
        
        # Shouldn't happen, just checking.
        values = subset.loc[:,"node2"]
        if(len(values) == 0):
            print("no values found for subset:\n{}\n".format(subset))
            print("row:\n{}\n".format(row))
        
        intervals_for_values = get_intervals_for_values_based_on_percentile(values, num_bins)
        
        lbounds = [pair[0] for pair in intervals_for_values]
        ubounds = [pair[1] for pair in intervals_for_values]
        df.loc[subset_mask,"lower_bound"] = lbounds
        df.loc[subset_mask,"upper_bound"] = ubounds
    
    df.to_csv(ail_file_out, sep = '\t', index = False)

    