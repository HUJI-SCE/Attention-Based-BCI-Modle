import pickle


class StateModelIndex:
    """
    This class represents the model state index:
    A simple counter value that keeps track of the index of the last training session
    of the network.

    How does it work:
    This index is used to name the files storing the network state.
    It is operated using the 'with' keyword in the following fashion:
    with StateModelIndex(filename) as state_model_index:

        ###
        Enter training code here
        Enter code storing model state here (using the value of state_model_index)
        Enter code updating the value of the state_model_index here
        ###

    at the end of the 'with' statement, the updated value of the state_model_index
    will be stored into the pickle file from whence it came.
    """

    def __init__(self, filename):
        self.filename = filename + '.pickle'
        self.value = None

    def __enter__(self):

        # Load the integer value from the pickle file
        try:
            with open(self.filename, 'rb') as file:
                self.value = pickle.load(file)

        except FileNotFoundError:
            # If the file doesn't exist or is empty, initialize value to 0
            # and store it into a pickle file
            self.value = 0
            with open(self.filename, 'wb') as file:
                pickle.dump(self.value, file)

        return self.value

    def __exit__(self, exc_type, exc_value, traceback):
        # Save the updated integer value to the pickle file
        with open(self.filename, 'wb') as file:
            pickle.dump(self.value, file)
