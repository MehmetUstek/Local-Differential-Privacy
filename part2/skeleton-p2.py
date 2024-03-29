from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import math
import random
import timeit

''' Globals '''
LABELS = ["", "frontpage", "news", "tech", "local", "opinion", "on-air", "misc", "weather", "msn-news", "health",
          "living", "business", "msn-sports", "sports", "summary", "bbs", "travel"]


def laplace_noise(x, scale):
    exp_eq = np.exp(- x / scale)
    return 1.0 / (2.0 * scale) * exp_eq

def average_limit_of_truncation(dataset):
    """
            Takes the histogram data, returns the average length of records. To appoint the optimal value for truncation.

            Args:
                dataset (list of lists): The MSNBC dataset

            Returns:
                Average length of records rounded to the nearest integer.
        """
    length_list = []
    for item in dataset:
        length_list.append(len(item))

    average = sum(length_list) / len(length_list)
    average = int(round(average))
    return average

""" 
    Helper functions
    (You can define your helper functions here.)
"""


def read_dataset(filename):
    """
        Reads the dataset with given filename.

        Args:
            filename (str): Path to the dataset file
        Returns:
            Dataset rows as a list of lists.
    """

    result = []
    with open(filename, "r") as f:
        for _ in range(7):
            next(f)
        for line in f:
            sequence = line.strip().split(" ")
            result.append([int(i) for i in sequence])
    return result


### HELPERS END ###


''' Functions to implement '''


def get_histogram(dataset: list):
    """
        Creates a histogram of given counts for each category and saves it to a file.

        Args:
            dataset (list of lists): The MSNBC dataset

        Returns:
            Ordered list of counts for each page category (frontpage, news, tech, ..., travel)
            Ex: [123, 383, 541, ..., 915]
    """
    counter = Counter()
    for row in dataset:
        for element in row:
            counter[LABELS[element]] += 1

    # plt.bar(counter.keys(), counter.values())
    # plt.ylabel('Counts')
    # plt.xticks(rotation='vertical', ha='center', va='center', wrap=True)
    # plt.savefig("np-histogram.png")
    # plt.show()
    list = []
    for item in counter.values():
        list.append(item)

    return list


def add_laplace_noise(real_answer: list, sensitivity: float, epsilon: float):
    """
        Adds laplace noise to query's real answers.

        Args:
            real_answer (list): Real answers of a query -> Ex: [92.85, 57.63, 12, ..., 15.40]
            sensitivity (float): Sensitivity
            epsilon (float): Privacy parameter
        Returns:
            Noisy answers as a list.
            Ex: [103.6, 61.8, 17.0, ..., 19.62]
    """
    scale = sensitivity / epsilon
    noisy_answer = []

    for item in real_answer:
        # noise = laplace_noise(item, scale)
        # noise = np.random.laplace(scale=scale)
        sample = np.random.default_rng().laplace(loc=0, scale=scale)
        noisy_answer.append(item + sample)

    return noisy_answer


def truncate(dataset: list, n: int):
    """
        Truncates dataset according to truncation parameter n.

        Args:  
            dataset: original dataset 
            n (int): truncation parameter
        Returns:
            truncated_dataset: truncated version of original dataset
    """
    truncated_dataset = []
    for item in dataset:
        truncated_dataset.append(item[:n])
    return truncated_dataset


def get_dp_histogram(dataset: list, n: int, epsilon: float):
    """
        Truncates dataset with parameter n and calculates differentially private histogram.

        Args:
            dataset (list of lists): The MSNBC dataset
            n (int): Truncation parameter
            epsilon (float): Privacy parameter
        Returns:
            Differentially private histogram as a list
    """
    truncated_dataset = truncate(dataset, n)
    counter = Counter()

    for row in truncated_dataset:
        for element in row:
            counter[LABELS[element]] += 1
    list = []
    for item in counter.values():
        list.append(item)
    noisy_hist = add_laplace_noise(list, 1.0, epsilon)
    # plt.bar(counter.keys(), noisy_hist)
    # plt.ylabel('Counts')
    # plt.xticks(rotation='vertical', ha='center', va='center', wrap=True)
    # plt.savefig("dp-histogram.png")
    # plt.show()


    return noisy_hist


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


def n_experiment(dataset, n_values: list, epsilon: float):
    """
        Function for the experiment explained in part (f).
        n_values is a list, such as: [1, 6, 11, 16 ...]
        Returns the errors as a list: [1256.6, 1653.5, ...] such that 1256.5 is the error when n=1,
        1653.5 is the error when n = 6, and so forth.
    """
    # timer_list = []
    total_errors = []
    non_private_histogram = get_histogram(dataset)
    for n in n_values:
        # start = timeit.default_timer()
        error_list = []
        for _ in range(30):
            dp_histogram = get_dp_histogram(dataset, n, epsilon)
            av_error = calculate_average_error(non_private_histogram, dp_histogram)
            error_list.append(av_error)

        total_average_error = sum(error_list) / len(error_list)

        total_errors.append(total_average_error)
        # Timers
        # stop = timeit.default_timer()
        # timer_list.append(stop-start)

    return total_errors


def epsilon_experiment(dataset, n: int, eps_values: list):
    """
        Function for the experiment explained in part (g).
        eps_values is a list, such as: [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
        Returns the errors as a list: [9786.5, 1234.5, ...] such that 9786.5 is the error when eps = 0.0001,
        1234.5 is the error when eps = 0.001, and so forth.
    """
    timer_list = []
    total_errors = []
    non_private_histogram = get_histogram(dataset)
    for epsilon in eps_values:
        start = timeit.default_timer()
        error_list = []
        for _ in range(30):
            dp_histogram = get_dp_histogram(dataset, n, epsilon)
            av_error = calculate_average_error(non_private_histogram, dp_histogram)
            error_list.append(av_error)

        total_average_error = sum(error_list) / len(error_list)

        total_errors.append(total_average_error)
        stop = timeit.default_timer()
        timer_list.append(stop-start)
    return total_errors, timer_list


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


def extract(dataset):
    """
        Extracts the first 1000 sequences and truncates them to n=1
    """
    extracted = dataset[0:1000]
    truncated_dataset = truncate(extracted, 1)
    return truncated_dataset


def most_visited_exponential(smaller_dataset, epsilon):
    """
        Using the Exponential mechanism, calculates private response for query: 
        "Which category (1-17) received the highest number of page visits?"

        Returns 1 for frontpage, 2 for news, 3 for tech, etc.
    """
    sensitivity = 1

    score = Counter()
    for row in smaller_dataset:
        for element in row:
            score[element] += 1

    probabilities = {}
    #
    denominator = 0.0

    for item in score.items():
        denominator += np.exp(epsilon * item[1] / 2 * sensitivity)
    for item in score.items():
        numerator = np.exp(epsilon * item[1] / 2 * sensitivity)
        probabilities[item[0]] = numerator / denominator
    choice = random.choices(list(probabilities.keys()),list(probabilities.values()))[0]

    return choice



def exponential_experiment(dataset, eps_values: list):
    """
        Function for the experiment explained in part (i).
        eps_values is a list such as: [0.001, 0.005, 0.01, 0.03, ..]
        Returns the list of accuracy results [0.51, 0.78, ...] where 0.51 is the accuracy when eps = 0.001,
        0.78 is the accuracy when eps = 0.005, and so forth.
    """
    dataset = extract(dataset)

    score = Counter()
    for row in dataset:
        for element in row:
            score[element] += 1
    correct_answer = score.most_common()[0][0]
    total_errors = []

    for epsilon in eps_values:
        correct_counter = 0
        for _ in range(1000):
            choice = most_visited_exponential(dataset, epsilon)
            # error_list.append(choice)
            # error_list.append(av_error)
            if choice == correct_answer:
                correct_counter += 1

        percentage = correct_counter / 1000.0
        total_errors.append(percentage)
    return total_errors


# FUNCTIONS TO IMPLEMENT END #

def main():
    dataset_filename = "msnbc.dat"
    dataset = read_dataset(dataset_filename)

    # Parts A, B, C, D, E
    non_private_histogram = get_histogram(dataset)
    print("Non private histogram:", non_private_histogram)
    calculated_trancation_parameter = average_limit_of_truncation(dataset)
    dp_histogram = get_dp_histogram(dataset, calculated_trancation_parameter, 0.01)
    print("DP histogram:", dp_histogram)
    av_error = calculate_average_error(non_private_histogram, dp_histogram)
    print("Average error:", av_error)

    # print("**** N EXPERIMENT RESULTS (f of Part 2) ****")
    # eps = 0.01
    # n_values = []
    # for i in range(1, 106, 5):
    #    n_values.append(i)
    # errors = n_experiment(dataset, n_values, eps)
    # for i in range(len(n_values)):
    #    print("n = ", n_values[i], " error = ", errors[i])
    #
    # print("*" * 50)

    ####### Analysis Of N values ########
    # For N vs Err Analysis
    # plt.plot(n_values, errors)
    # plt.ylabel('Errors')
    # plt.xlabel('N values')
    # plt.savefig("NvsErr_plot.png")
    # plt.show()

    # For N vs Computational Time Analysis
    # plt.plot(n_values, timer_list)
    # plt.ylabel('Computational Time')
    # plt.xlabel('N values')
    # plt.savefig("NvsComputationalTime_plot.png")
    # plt.show()
    ####### Analysis ########


    # print("**** EPSILON EXPERIMENT RESULTS (g of Part 2) ****")
    # n = 50
    # eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    # errors, timer_list = epsilon_experiment(dataset, n, eps_values)
    # for i in range(len(eps_values)):
    #    print("eps = ", eps_values[i], " error = ", errors[i])
    #
    # print("*" * 50)

    ####### Analysis Of N values ########
    # For Eps vs Err Analysis for epsilon exp.
    # plt.plot(eps_values, errors)
    # plt.ylabel('Errors')
    # plt.xlabel('Eps values')
    # plt.savefig("EpsvsErr_plot_.png")
    # plt.show()
    #
    # # For N vs Computational Time Analysis
    # plt.plot(eps_values, timer_list)
    # plt.ylabel('Computational Time')
    # plt.xlabel('Eps values')
    # plt.savefig("EpsvsComputationalTime_plot.png")
    # plt.show()
    ####### Analysis ########


    # print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    # eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    # exponential_experiment_result = exponential_experiment(dataset, eps_values)
    # for i in range(len(eps_values)):
    #    print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])

    ####### Analysis Of N values ########
    # For Eps vs Accuracy Analysis for exponential exp.
    # plt.plot(eps_values, exponential_experiment_result)
    # plt.ylabel('Accuracy')
    # plt.xlabel('Eps values')
    # plt.savefig("EpsvsAccuracy_plot_.png")
    # plt.show()
    ####### Analysis ########

if __name__ == "__main__":
    main()
