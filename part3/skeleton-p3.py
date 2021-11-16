import math, random


""" Globals """
DOMAIN = list(range(25)) # [0, 1, ..., 24]

""" Helpers """

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

# TODO: Implement this function!
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
    pass

# TODO: Implement this function!
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
    pass

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
    return 0.0

# TODO: Implement this function!
def encode_rappor(val):
    """
        Encodes the given value into a bit vector.

        Args:
            val (int): The user's true value.
        Returns:
            The encoded bit vector as a list: [0, 1, ..., 0]
    """
    pass

# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    """
        Perturbs the given bit vector using RAPPOR protocol.

        Args:
            encoded_val (list) : User's encoded value
            epsilon (float): Privacy parameter
        Returns:
            Perturbed bit vector that the user reports to the server as a list: [1, 1, ..., 0]
    """
    pass

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
    pass
    
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
    return 0.0

def main():
    dataset = read_dataset("daily_time.txt")
    
    print("GRR EXPERIMENT")
    #for epsilon in [20.0]: 
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]: 
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)
    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))
    

if __name__ == "__main__":
    main()