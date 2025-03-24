from collections import OrderedDict

class Experiment:
    """
    Represents an experiment, focusing on experiment setup and sample management.

    Attributes:
        n_conditions (int): The number of conditions in the experiment.
        conditions_list (list[str]): Labels for each condition in the experiment.
        number_of_samples_list (list[int]): The number of samples for each condition.
        aggregate_number_of_samples_list (list[int]): The cumulative sum of samples up to each condition.
        extensive_conditions_list (list[str]): A flat list replicating condition labels for each sample.
        individual_samples_list (list[list[str]]): Lists of samples grouped by each condition.
        full_samples_list (list[str]): A list of all sample labels in the experiment.
    """

    def __init__(self):
        """
        Initializes the Experiment with default values for its attributes.
        """
        self.n_conditions = 0
        self.conditions_list = []
        self.number_of_samples_list = []
        self.aggregate_number_of_samples_list = []
        self.extensive_conditions_list = []
        self.individual_samples_list = []
        self.full_samples_list = []

    def setup_experiment(self, n_conditions, conditions_list, number_of_samples_list):
        """
        Sets up the experiment's attributes based on provided parameters.

        Parameters:
            n_conditions (int): The number of conditions in the experiment.
            conditions_list (list[str]): The labels for each condition.
            number_of_samples_list (list[int]): The number of samples for each condition.

        Returns:
            bool: True if the setup is successful, False otherwise.
        """
        self.n_conditions = n_conditions
        self.conditions_list = conditions_list
        self.number_of_samples_list = number_of_samples_list

        # Validate the conditions' labels for emptiness
        if not self.validate_conditions():
            return False

        # Generate various lists based on the given conditions and samples
        self.generate_full_samples_list()
        self.calculate_aggregate_samples()
        self.generate_individual_samples_list()
        self.generate_extensive_conditions_list()
        return True

    def validate_conditions(self):
        """
        Validates the conditions_list for any empty labels.

        Returns:
            bool: True if all labels are valid, False otherwise.
        """
        return all(label != "" for label in self.conditions_list)

    def generate_full_samples_list(self):
        """
        Generates a list of sample labels for the experiment based on the total number of samples.
        """
        self.full_samples_list = [f's{i+1}' for i in range(sum(self.number_of_samples_list))]

    def calculate_aggregate_samples(self):
        """
        Calculates the cumulative sum of samples for each condition.
        """
        self.aggregate_number_of_samples_list = [sum(self.number_of_samples_list[:i+1]) for i in range(len(self.number_of_samples_list))]

    def generate_individual_samples_list(self):
        """
        Generates sublists of samples for each condition.
        """
        self.individual_samples_list = []  # Clear existing list contents
        start_index = 0
        for num_samples in self.number_of_samples_list:
            end_index = start_index + num_samples
            self.individual_samples_list.append(self.full_samples_list[start_index:end_index])
            start_index = end_index

    def generate_extensive_conditions_list(self):
        """
        Creates a flat list replicating condition labels for each sample.
        """
        self.extensive_conditions_list = [condition for condition, num_samples in zip(self.conditions_list, self.number_of_samples_list) for _ in range(num_samples)]

    def update_dataframe(self, bad_samples, dataframe):
        """
        Removes columns corresponding to bad samples from the DataFrame.

        Parameters:
            bad_samples (list[str]): A list of sample labels considered as bad.
            dataframe (pd.DataFrame): The DataFrame to be updated.

        Returns:
            pd.DataFrame: The updated DataFrame with bad sample columns removed.
        """
        for sample in bad_samples:
            col_name = f'concentration[{sample}]'
            if col_name in dataframe.columns:
                dataframe.drop(columns=[col_name], inplace=True)
        return dataframe

    def update_full_samples_list(self, bad_samples):
        """
        Removes bad samples from the full_samples_list.

        Parameters:
            bad_samples (list[str]): A list of sample labels considered as bad.
        """
        self.full_samples_list = [sample for sample in self.full_samples_list if sample not in bad_samples]

    def rebuild_extensive_conditions_list(self):
        """
        Rebuilds the extensive_conditions_list based on the updated full_samples_list.

        This method is typically called after removing bad samples to ensure the 
        extensive_conditions_list aligns with the current state of full_samples_list.
        """
        updated_list = []
        for sample in self.full_samples_list:
            original_index = self.full_samples_list_before_removal.index(sample)
            updated_list.append(self.extensive_conditions_list_before_removal[original_index])
        self.extensive_conditions_list = updated_list

    def update_conditions_and_samples(self):
        """
        Updates the conditions_list and number_of_samples_list based on the current 
        extensive_conditions_list.
    
        This method realigns the conditions and their respective sample counts 
        after any modification to the sample lists, ensuring the order of conditions is maintained.
        """
        # Using an OrderedDict to maintain the order of conditions
        conditions_order = OrderedDict()
        for condition in self.extensive_conditions_list:
            if condition not in conditions_order:
                conditions_order[condition] = 1
            else:
                conditions_order[condition] += 1
    
        # Updating conditions_list with the keys of the OrderedDict
        self.conditions_list = list(conditions_order.keys())
        # Updating number_of_samples_list with the counts of each condition
        self.number_of_samples_list = list(conditions_order.values())


    def recalculate_aggregate_samples_list(self):
        """
        Recalculates the aggregate_number_of_samples_list to reflect current 
        conditions and sample counts.

        This method is important after any modification to the experiment's 
        setup, ensuring accurate tracking of cumulative samples.
        """
        self.aggregate_number_of_samples_list = [sum(self.number_of_samples_list[:i+1]) for i in range(len(self.number_of_samples_list))]

    def remove_bad_samples(self, bad_samples, df):
        """
        Integrates steps to remove specified bad samples and updates the experiment's attributes.
        
        Parameters:
            bad_samples (list[str]): A list of sample labels considered as bad.
            df (pd.DataFrame): The DataFrame to be updated.
        
        Returns:
            pd.DataFrame: The updated DataFrame with bad sample columns removed.
        """
        # Backup current state before removal
        self.full_samples_list_before_removal = self.full_samples_list.copy()
        self.extensive_conditions_list_before_removal = self.extensive_conditions_list.copy()
    
        # Remove bad samples from the DataFrame
        for sample in bad_samples:
            mean_area_col = f'concentration[{sample}]'
            if mean_area_col in df.columns:
                df = df.drop(columns=[mean_area_col])
    
        # Update internal lists
        self.full_samples_list = [sample for sample in self.full_samples_list if sample not in bad_samples]
        self.rebuild_extensive_conditions_list()
        self.update_conditions_and_samples()
        self.recalculate_aggregate_samples_list()
        self.generate_individual_samples_list()
        self.n_conditions = len(self.conditions_list)
    
        # Remove conditions with no samples
        self.conditions_list = [cond for cond, samples in zip(self.conditions_list, self.individual_samples_list) if samples]
        self.individual_samples_list = [samples for samples in self.individual_samples_list if samples]
        self.number_of_samples_list = [len(samples) for samples in self.individual_samples_list]
    
        return df

    def __repr__(self):
        """
        Represents the Experiment object as a string for easy readability.

        Returns:
            str: A string representation of the Experiment object.
        """
        return (f"conditions_list: {self.conditions_list}, "
                f"number_of_samples_list: {self.number_of_samples_list}, "
                f"aggregate_number_of_samples_list: {self.aggregate_number_of_samples_list}, "
                f"extensive_conditions_list: {self.extensive_conditions_list}, "
                f"individual_samples_list: {self.individual_samples_list}, "
                f"full_samples_list: {self.full_samples_list}")