import matplotlib.pyplot as plt
import numpy as np
import math
import random

''' Globals '''
LABELS = ["", "frontpage", "news", "tech", "local", "opinion", "on-air", "misc", "weather", "msn-news", "health", "living", "business", "msn-sports", "sports", "summary", "bbs", "travel"]


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

# TODO: Implement this function!
def get_histogram(dataset: list):
    """
        Creates a histogram of given counts for each category and saves it to a file.

        Args:
            dataset (list of lists): The MSNBC dataset

        Returns:
            Ordered list of counts for each page category (frontpage, news, tech, ..., travel)
            Ex: [123, 383, 541, ..., 915]
    """
    pass


# TODO: Implement this function!
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
    pass


# TODO: Implement this function!
def truncate(dataset: list, n: int):
    """
        Truncates dataset according to truncation parameter n.

        Args:  
            dataset: original dataset 
            n (int): truncation parameter
        Returns:
            truncated_dataset: truncated version of original dataset
    """
    pass


# TODO: Implement this function!
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
    pass


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):
    """
        Calculates error according to the equation stated in part (e).

        Args: Actual histogram (list), Noisy histogram (list)
        Returns: Error (Err) in the noisy histogram (float)
    """
    pass


# TODO: Implement this function!
def n_experiment(dataset, n_values: list, epsilon: float):
    """
        Function for the experiment explained in part (f).
        n_values is a list, such as: [1, 6, 11, 16 ...]
        Returns the errors as a list: [1256.6, 1653.5, ...] such that 1256.5 is the error when n=1,
        1653.5 is the error when n = 6, and so forth.
    """
    return []


# TODO: Implement this function!
def epsilon_experiment(dataset, n: int, eps_values: list):
    """
        Function for the experiment explained in part (g).
        eps_values is a list, such as: [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
        Returns the errors as a list: [9786.5, 1234.5, ...] such that 9786.5 is the error when eps = 0.0001,
        1234.5 is the error when eps = 0.001, and so forth.
    """
    return []


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def extract(dataset):
    """
        Extracts the first 1000 sequences and truncates them to n=1
    """
    pass


# TODO: Implement this function!
def most_visited_exponential(smaller_dataset, epsilon):
    """
        Using the Exponential mechanism, calculates private response for query: 
        "Which category (1-17) received the highest number of page visits?"

        Returns 1 for frontpage, 2 for news, 3 for tech, etc.
    """
    pass


# TODO: Implement this function!
def exponential_experiment(dataset, eps_values: list):
    """
        Function for the experiment explained in part (i).
        eps_values is a list such as: [0.001, 0.005, 0.01, 0.03, ..]
        Returns the list of accuracy results [0.51, 0.78, ...] where 0.51 is the accuracy when eps = 0.001,
        0.78 is the accuracy when eps = 0.005, and so forth.
    """
    return []


# FUNCTIONS TO IMPLEMENT END #

def main():
    dataset_filename = "msnbc.dat"
    dataset = read_dataset(dataset_filename)

    #non_private_histogram = get_histogram(dataset)
    #print("Non private histogram:", non_private_histogram)
    
    #print("**** N EXPERIMENT RESULTS (f of Part 2) ****")
    #eps = 0.01
    #n_values = []
    #for i in range(1, 106, 5):
    #    n_values.append(i)
    #errors = n_experiment(dataset, n_values, eps)
    #for i in range(len(n_values)):
    #    print("n = ", n_values[i], " error = ", errors[i])   
    
    print("*" * 50)

    #print("**** EPSILON EXPERIMENT RESULTS (g of Part 2) ****")    
    #n = 50
    #eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    #errors = epsilon_experiment(dataset, n, eps_values)
    #for i in range(len(eps_values)):
    #    print("eps = ", eps_values[i], " error = ", errors[i])
    
    print("*" * 50)

    #print ("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    #eps_values = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    #exponential_experiment_result = exponential_experiment(dataset, eps_values)
    #for i in range(len(eps_values)):
    #    print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])    


if __name__ == "__main__":
    main()
