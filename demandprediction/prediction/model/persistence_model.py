import logging
from typing import Dict, Union
import tensorflow as tf
import pandas as pd

from demandprediction.prediction.generators.border_generator import BorderDataHandler
from demandprediction.prediction.model import BaseModel
from demandprediction.prediction.postprocessing.error_measures import log_grid_error
from demandprediction.prediction.postprocessing.postprocessing import calculate_grid_error
from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor
from demandprediction.utils.shared_filenames import get_predictions_filename

class PersistenceModel(BaseModel):
    def __init__(self, config):
        """
        Initializes the Persitance model.
        :param config: Configuration for the prediction.
        """
        super().__init__(model_name=type(self).__name__, data_directory=config['data_directory'])
        self.start_date = config['start_date']
        self.stop_date = config['stop_date']
        self.number_of_grid_rows = config['number_of_grid_rows']
        self.number_of_grid_columns = config['number_of_grid_columns']
        self.lags = config['lags']
        self.step_size = config['step_size']
        self.x_limits = (config['x_min'], config['x_max'])
        self.y_limits = (config['y_min'], config['y_max'])
        self.border_width = config['border_width']
        prediction_relative_path = get_predictions_filename(self.model_name, self.start_time)
        self.prediction_filepath = self.data_directory.joinpath(prediction_relative_path)
        self.number_of_prediction_timesteps = 0
        self.border_models: Dict[str, Union[None, tf.keras.models.Model]] = {
            'north': None,
            'east': None,
            'south': None,
            'west': None
        }
        self.grid_predictions = None
        self.border_predictions = None
        self.data_handler: Union[None, BorderDataHandler] = None
        self.trained = True

        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        super().prepare_data()
        data_preprocessor = DataPreprocessor(self.data_directory, self.x_limits, self.y_limits)
        # Generate grid and border data
        self.ground_truth_filepath = data_preprocessor.generate_grid_with_border(
            stop_date=self.stop_date,
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            step_size=self.step_size,
            feature_name='demand',
            border_width=self.border_width
        )

        # set flag
        self.data_prepared = True

    def generate_model(self):
        self.logger.info("Model generation isn´t needed. Just use predict.")


    def train_model(self, plot_fitting_history=None, save_model=True):
        self.logger.info("Model training isn´t needed. Just use predict.")

    def predict(self):
        super().predict()

        grid_data = pd.read_hdf(self.ground_truth_filepath, key='even_grid')  # type: pd.DataFrame
        border_data = pd.read_hdf(self.ground_truth_filepath, key='border')  # type: pd.DataFrame


        self.grid_predictions = grid_data.shift(1)[self.start_date:self.stop_date]
        self.border_predictions = border_data.shift(1)[self.start_date:self.stop_date]
        self.predictions = True
        self.logger.info('Prediction successful')

    def save_predictions(self):
        super().save_predictions()

        grid_predictions_df = self.grid_predictions

        border_predictions_df = self.border_predictions

        grid_predictions_df.to_hdf(self.prediction_filepath, key='grid_predictions', complevel=6)
        border_predictions_df.to_hdf(self.prediction_filepath, key='border_predictions', complevel=6)

    def score_model(self):
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
            logger=self.logger,
            actual_df_key='even_grid',
            prediction_df_key='grid_predictions'
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
            'step_size': [self.step_size],
        })

        filename = self.data_directory.joinpath("evaluation/metrics_and_hyperparams.csv")
        with open(filename, 'a') as f:
            df.to_csv(filename, mode='a', header=not f.tell())

        log_grid_error(self.logger, self.prediction_filepath, errors)

        self.logger.info("Model errors were calculated successfully")
