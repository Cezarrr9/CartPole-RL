import numpy as np

def bucketize(obs: np.ndarray, obs_bounds: list) -> tuple:
    """ Discretizes the continuous state values into a fixed
    number of buckets.

    Parameters:
    - obs (np.ndarray): The continuous observation given by the environment.
    - obs_bounds (list[tuple]): The accepted boundaries of the observation space for each type
    of information (check https://gymnasium.farama.org/environments/classic_control/cart_pole/#observation-space).

    Returns:
    - bucket_indices (tuple[int]): The indices of the buckets for each type of information.
    
    """
    # Define the number of buckets for each type of observation.
    n_buckets = (1, 1, 6, 3)
    
    # Initialize a list to hold the bucket indices for each observation.
    bucket_indices = []

    for i in range(len(obs)):
        
        # If the value is smaller than or equal to the inferior boundary,
        # then the it is associated with the first bucket
        if obs[i] <= obs_bounds[i][0]:
            bucket_index = 0

        # If the value is bigger than or equal to the superior boundary, 
        # then it is associated with the last bucket
        elif obs[i] >= obs_bounds[i][1]:
            bucket_index = n_buckets[i] - 1

        else:
            # Compute the range of the observation's boundaries
            bound_width = obs_bounds[i][1] - obs_bounds[i][0]

            # Compute the offset based on the inferior
            # boundary and the number of buckets 
            offset = (n_buckets[i] - 1) * obs_bounds[i][0] / bound_width

            # Compute the scaling factor to adjust the observation
            # within the bucket range
            scaling = (n_buckets[i] - 1) / bound_width

            # Apply the scaling and offset to determine
            # the bucket index, rounding to the nearest integer
            bucket_index = int(round(scaling * obs[i] - offset))

        # Store the bucket index 
        bucket_indices.append(bucket_index)
    
    bucket_indices = tuple(bucket_indices)
    return bucket_indices