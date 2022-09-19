import logging
import os

import pandas as pd

from demandprediction.prediction.model import BaseModel
from demandprediction.prediction.postprocessing.error_measures import log_grid_error
from demandprediction.prediction.postprocessing.postprocessing import calculate_grid_error
from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor
from demandprediction.utils.shared_filenames import get_predictions_filename


class ConvLSTMBaseModel(BaseModel):
    def __init__(self, model_name: str, config: dict):
        """
        Initializes the ConvLSTM model.
        Inputs are: Demand and Metainformation
        The Architecture does use a Hadamard layer
        :param config: Configuration for the prediction.
        """
        super().__init__(model_name, config['data_directory'])
        self.start_date = config['start_date']
        self.stop_date = config['stop_date']
        self.number_of_grid_rows = config['number_of_grid_rows']
        self.number_of_grid_columns = config['number_of_grid_columns']
        self.lags = config['lags']
        self.number_of_filters = config['number_of_filters']
        self.kernel_size = config['kernel_size']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.step_size = config['step_size']
        self.epochs = config['epochs']
        self.x_limits = (config['x_min'], config['x_max'])
        self.y_limits = (config['y_min'], config['y_max'])
        prediction_relative_path = get_predictions_filename(self.model_name, self.start_time)
        self.prediction_filepath = os.path.join(self.data_directory, prediction_relative_path)
        self.number_of_prediction_timesteps = 0

        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        super().prepare_data()
        data_preprocessor = DataPreprocessor(self.data_directory, self.x_limits, self.y_limits)
        self.ground_truth_filepath = data_preprocessor.generate_window_grid(
            stop_date=self.stop_date,
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            step_size=self.step_size,
            feature_name='demand'
        )

        self.number_of_prediction_timesteps = len(
            DataPreprocessor.get_date_range_between_dates(
                start_date=self.start_date,
                end_date=self.stop_date,
                freq=f'{self.step_size}min',
                end_inclusive=False
            )
        )
        # set flag
        self.data_prepared = True

    def save_predictions(self):
        super().save_predictions()
        predictions = self.predictions.reshape(self.predictions.shape[0],
                                               self.number_of_grid_rows * self.number_of_grid_columns)
        predictions_df = pd.DataFrame(predictions)

        # load actual data for index
        # noinspection PyTypeChecker
        actual_data_df = pd.read_hdf(self.ground_truth_filepath, key='even_grid')  # type: pd.DataFrame
        predictions_df.index = actual_data_df.index[-len(predictions_df.index):]
        self.logger.debug(predictions_df.index[:10])
        self.logger.debug(predictions_df.index[-10:])

        predictions_df.to_hdf(self.prediction_filepath, key='predictions', complevel=6)
        self.logger.info(f"Predictions saved to file: {self.prediction_filepath}")

    def score_model(self):
        """
        Calculate the evaluation metrics (RMSE, MAPE, etc.).
        Save the metrics and model hyperparams to csv file.
        """
        super().score_model()

        # Calculate error
        model_id = "{}_{date:%Y-%m-%d_%H-%M-%S}.h5".format(
            self.model_name, date=self.start_time)

        errors = calculate_grid_error(
            filepath=self.ground_truth_filepath,
            prediction_filepath=self.prediction_filepath,
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            ceiling=1,
            logger=self.logger
        )

        # Write the errors and all hyperparams to a csv file!
        df = pd.DataFrame({
            'model_id': [model_id],
            'model_name': [self.model_name],
            'avg_grid_rmse_above_ceiling': [errors[0]],
            'avg_grid_mae_above_ceiling': [errors[1]],
            'avg_grid_mapec_above_ceiling': [errors[2]],
            'global_rmse': [errors[3]],
            'global_mae': [errors[4]],
            'global_mape': [errors[5]],
            'number_of_rows': [self.number_of_grid_rows],
            'number_of_columns': [self.number_of_grid_columns],
            'lags': [self.lags],
            'nb_filters': [self.number_of_filters],
            'k_size': [self.kernel_size],
            'batch_size': [self.batch_size],
            'lr': [self.learning_rate],
            'step_size': [self.step_size],
        })

        filename = os.path.join(self.data_directory, "evaluation/metrics_and_hyperparams.csv")
        with open(filename, 'a') as f:
            df.to_csv(filename, mode='a', header=not f.tell())

        log_grid_error(self.logger, self.prediction_filepath, errors)

        self.logger.info("Model errors were calculated successfully")
