import math, random
from collections import Counter

import numpy as np


""" Globals """
DOMAIN = list(range(25)) # [0, 1, ..., 24]

""" Helpers """
def calculate_average_error(actual_hist, noisy_hist):
    """
        Calculates error according to the equation stated in part (e).

        Args: Actual histogram (list), Noisy histogram (list)
        Returns: Error (Err) in the noisy histogram (float)
    """
    error = 0.0
    for i in range(len(actual_hist)):
        error += abs(actual_hist[i] - noisy_hist[i])

    error = error / len(actual_hist)
    return error

def read_dataset(filename):
    """
        Reads the dataset with given filename.

        Args:
            filename (str): Path to the dataset file
        Returns:
            Dataset rows as a list.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result

# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """

def perturb_grr(val, epsilon):
    """
        Perturbs the given value using GRR protocol.

        Args:
            val (int): User's true value
            epsilon (float): Privacy parameter
        Returns:
            Perturbed value that the user reports to the server
    """
    d = len(DOMAIN)
    probability_of_truth = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    if random.random() <= probability_of_truth:
        return val
    else:
        domain_except_val = DOMAIN.copy()
        domain_except_val.remove(val)
        return random.sample(domain_except_val,1)[0]


def estimate_grr(perturbed_values, epsilon):
    """
        Estimates the histogram given GRR perturbed values of the users.

        Args:
            perturbed_values (list): Perturbed values of all users
            epsilon (float): Privacy parameter
        Returns:
            Estimated histogram as a list: [1.5, 6.7, ..., 1061.0] 
            for each hour in the domain [0, 1, ..., 24] respectively.
    """
    d = len(DOMAIN)
    total_n = len(perturbed_values)
    counter = Counter()
    for item in perturbed_values:
        counter[item] += 1
    # for item in perturbed_values:
    probability_of_truth = np.exp(epsilon) / (np.exp(epsilon) + d - 1)
    probability_of_lie = 1 - probability_of_truth / (np.exp(epsilon) + d - 1)
    cv_list = []
    for item in DOMAIN:
        nv = counter[item]
        Iv = probability_of_truth * nv + (total_n-nv) * probability_of_lie
        cv = (Iv - total_n * probability_of_lie) / (probability_of_truth - probability_of_lie)
        cv_list.append(cv)
    return cv_list


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    """
        Conducts the data collection experiment for GRR.

        Args:
            dataset (list): The daily_time dataset
            epsilon (float): Privacy parameter
        Returns:
            Error of the estimated histogram (float) -> Ex: 330.78
    """
    real_counter = Counter()
    real_list = []
    for item in dataset:
        real_counter[item] += 1
    for item in DOMAIN:
        real_list.append(real_counter[item])
    perturbed_dataset = []
    for item in dataset:
        perturbed_dataset.append(perturb_grr(item, epsilon))
    estimated_list = estimate_grr(perturbed_dataset,epsilon)

    error = calculate_average_error(real_list, estimated_list)
    return error


def encode_rappor(val):
    """
        Encodes the given value into a bit vector.

        Args:
            val (int): The user's true value.
        Returns:
            The encoded bit vector as a list: [0, 1, ..., 0]
    """
    bitvector = [0] * 25
    bitvector[val] = 1
    return bitvector


def perturb_rappor(encoded_val, epsilon):
    """
        Perturbs the given bit vector using RAPPOR protocol.

        Args:
            encoded_val (list) : User's encoded value
            epsilon (float): Privacy parameter
        Returns:
            Perturbed bit vector that the user reports to the server as a list: [1, 1, ..., 0]
    """
    preserve_bit_prob = np.exp(epsilon / 2) / (np.exp(epsilon/2) + 1)
    flip_bit_prob = 1 / (np.exp(epsilon/2) + 1)
    bit_iterator = 0
    for bit in encoded_val:
        if random.random() <= preserve_bit_prob:
            bit_iterator += 1
            continue
        else:
            encoded_val[bit_iterator] = int(not bit)
            bit_iterator += 1
    return encoded_val


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):
    """
        Estimates the histogram given RAPPOR perturbed values of the users.

        Args:
            perturbed_values (list of lists): Perturbed bit vectors of all users
            epsilon (float): Privacy parameter
        Returns:
            Estimated histogram as a list: [1.5, 6.7, ..., 1061.0] 
            for each hour in the domain [0, 1, ..., 24] respectively.
    """
    total_n = len(perturbed_values)
    bit_counter = Counter()
    for user in perturbed_values:
        domain_iterator = 0
        for bit in user:
            if bit:
                bit_counter[domain_iterator] += 1
                domain_iterator += 1
    probability_of_truth = np.exp(epsilon/2) / (np.exp(epsilon/2) + 1)
    probability_of_lie = 1 / (np.exp(epsilon/2) + 1)
    cv_list = []
    for item in DOMAIN:
        nv = bit_counter[item]
        Iv = probability_of_truth * nv + (total_n - nv) * probability_of_lie
        cv = (Iv - total_n * probability_of_lie) / (probability_of_truth - probability_of_lie)
        cv_list.append(cv)
    return cv_list
    
# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):
    """
        Conducts the data collection experiment for RAPPOR.

        Args:
            dataset (list): The daily_time dataset
            epsilon (float): Privacy parameter
        Returns:
            Error of the estimated histogram (float) -> Ex: 330.78
    """
    # real_counter = Counter()
    # real_list = []
    # for item in dataset:
    #     real_counter[item] += 1
    # for item in DOMAIN:
    #     real_list.append(real_counter[item])
    perturbed_dataset_rappor = []
    for user_val in dataset:
        encoded = encode_rappor(user_val)
        perturbed_dataset_rappor.append(perturb_rappor(encoded,epsilon))
    estimated_list = estimate_rappor(perturbed_dataset_rappor,epsilon=epsilon)
    # error = calculate_average_error(real_list, estimated_list)
    # return error
    pass

def main():
    dataset = read_dataset("daily_time.txt")
    perturbed_dataset = []
    for item in dataset:
        perturbed_dataset.append(perturb_grr(item,4.0))

    # print("GRR EXPERIMENT")
    # #for epsilon in [20.0]:
    # for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
    #     error = grr_experiment(dataset, epsilon)
    #     print("e={}, Error: {:.2f}".format(epsilon, error))

    # print("*" * 50)
    epsilon = 0.01
    perturbed_dataset_rappor = []
    for user_val in dataset:
        encoded = encode_rappor(user_val)
        perturbed_dataset_rappor.append(perturb_rappor(encoded,epsilon))
    cv_list = estimate_rappor(perturbed_dataset_rappor,epsilon=epsilon)
    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))
    

if __name__ == "__main__":
    main()