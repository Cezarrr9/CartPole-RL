def bucketize(obs, obs_bounds):
    n_buckets = (1, 1, 6, 3)
    bucket_indices = []
    for i in range(len(obs)):
        if obs[i] <= obs_bounds[i][0]:
            bucket_index = 0
        elif obs[i] >= obs[i][1]:
            bucket_index = n_buckets[i] - 1
        else:
            bound_width = obs_bounds[i][1] - obs_bounds[i][0]
            offset = (n_buckets[i] - 1) * obs_bounds[i][0] / bound_width
            scaling = (n_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * obs[i] - offset))
 
        bucket_indices.append(bucket_index)
    
    return tuple(bucket_indices)