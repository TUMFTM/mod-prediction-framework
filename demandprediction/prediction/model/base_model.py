__author__ = 'Maximilian Speicher'
__email__ = "maximilian.speicher@tum.de"
__status__ = "Development"

import datetime
import logging
from abc import ABC, abstractmethod


class BaseModel(ABC):

    def __init__(self, model_name, data_directory):
        """
        Initializes a model.
        :param model_name: Name of the model.
        """
        # Start time as unique id
        self.start_time = datetime.datetime.now()
        self.logger = logging.getLogger(__name__)

        # save model name
        self.model_name = model_name

        # initialize later needed attributes
        self.ground_truth_filepath = None
        self.model = None
        self.predictions = None
        self.trained = False
        self.data_prepared = False

        self.data_directory = data_directory
        self.logger.info(f'Successfully instantiated {model_name}')

    @abstractmethod
    def prepare_data(self):
        """
        Prepares the needed data.
        """
        self.logger.info('Data preparation started.\n')
        pass

    @abstractmethod
    def generate_model(self):
        """
        Generates the model. (If needed)
        """
        if self.model is not None:
            self.logger.info('Model generated succesfully:\n')
        pass

    @abstractmethod
    def train_model(self, plot_fitting_history=None):
        """
        Trains the model. (If needed)
        :param plot_fitting_history: If set to True plots the training history for neural networks.
        """
        if self.model is None:
            raise ModelNotGenerated('No Model has been generated so far')
        if not self.data_prepared:
            raise NoDataPreparedError('The data hasn´t been prepared so far')
        self.logger.info('Training of the model started.\n')
        pass

    @abstractmethod
    def predict(self):
        """
        Make predictions.
        """
        if not self.trained:
            raise ModelNotTrainedError('The model hasn´t been trained so far')
        self.logger.info('Prediction started.\n')
        pass

    @abstractmethod
    def save_predictions(self):
        """
        Save the predictions to ../data/predictions/model_name_start_time.h5
        """
        if self.predictions is None:
            raise NoPredictionGeneratedError('No predictions have been made so far')
        self.logger.info('Predictions of the model {} are being saved'.format(self.model_name))

    @abstractmethod
    def score_model(self):
        """
        Calculates error of the prediction.
        :return: The calculated error
        """
        if self.predictions is None:
            raise NoPredictionGeneratedError('No predictions have been made so far')


class ModelNotGenerated(Exception):
    pass


class ModelNotTrainedError(Exception):
    pass


class NoDataPreparedError(Exception):
    pass


class NoPredictionGeneratedError(Exception):
    pass
