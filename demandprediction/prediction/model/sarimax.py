import logging
import concurrent.futures

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from demandprediction.prediction.model import BaseModel
from demandprediction.prediction.postprocessing.error_measures import log_grid_error
from demandprediction.prediction.postprocessing.postprocessing import calculate_grid_error
from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor
from demandprediction.utils.shared_filenames import get_predictions_filename, get_score_filename


class SARIMAXModel(BaseModel):
    def __init__(self, config: dict, order: tuple, seasonal_order: tuple, trend: str, max_workers: int):
        """
        Initializes the naive model.
        :param config: Configuration for the prediction.
        """
        super().__init__(model_name=type(self).__name__, data_directory=config['data_directory'])
        self.start_date = config['start_date']
        self.stop_date = config['stop_date']
        self.number_of_grid_rows = config['number_of_grid_rows']
        self.number_of_grid_columns = config['number_of_grid_columns']
        self.step_size = config['step_size']
        self.x_limits = (config['x_min'], config['x_max'])
        self.y_limits = (config['y_min'], config['y_max'])
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.max_workers = max_workers

        self.data = None

        self.number_of_timesteps_per_model = 4 * 24 * 8
        self.number_of_prediction_timesteps = len(
            DataPreprocessor.get_date_range_between_dates(
                start_date=self.start_date,
                end_date=self.stop_date,
                freq=f'{self.step_size}min',
                end_inclusive=False
            )
        )

        prediction_relative_path = get_predictions_filename(self.model_name, self.start_time)
        self.prediction_filepath = self.data_directory.joinpath(prediction_relative_path)

        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        """
        Generates the citywide timeseries and saves it to the disk.
        """
        super().prepare_data()
        # Create DataPreprocessor object and generate citywide timeseries
        data_preprocessor = DataPreprocessor(self.data_directory, self.x_limits, self.y_limits)
        self.ground_truth_filepath = data_preprocessor.generate_window_grid(
            stop_date=self.stop_date,
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            step_size=self.step_size,
            feature_name='demand'
        )
        # set flag
        self.data_prepared = True

    def generate_model(self):
        self.logger.info("Model generation isn´t needed. Just use predict.")

    def train_model(self, plot_fitting_history=None):
        self.logger.info("Model training isn´t needed. Just use predict.")

    # noinspection PyTypeChecker
    def predict(self):
        list_of_gridcells = self.__generate_list_of_cells()

        # load data once to minimize disk usage (bottleneck)
        self.data = pd.read_hdf(self.ground_truth_filepath, key='even_grid')  # type: pd.DataFrame

        list_of_prediction_dfs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for gridcell in list_of_gridcells:
                future_to_prediction = [executor.submit(predict_for_gridcell, gridcell, self.data[gridcell],
                                                        self.number_of_prediction_timesteps, self.order,
                                                        self.seasonal_order, self.trend,
                                                        self.number_of_timesteps_per_model)]

            for future in concurrent.futures.as_completed(future_to_prediction):
                list_of_prediction_dfs.append(future.result())

        predictions = pd.concat(list_of_prediction_dfs, axis=1)
        predictions = predictions.reindex(sorted(predictions.columns), axis=1)

        self.predictions = predictions
        self.logger.info('Prediction successful')

    def save_predictions(self):
        super().save_predictions()
        predictions_df = pd.DataFrame(self.predictions)

        # get index for predictions
        predictions_df.index = self.data.index[-len(predictions_df.index):]

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
            'step_size': [self.step_size],
        })

        filename = self.data_directory.joinpath(get_score_filename(self.model_name, self.start_time))
        with open(filename, 'a') as f:
            df.to_csv(filename, mode='a', header=not f.tell())

        log_grid_error(self.logger, self.prediction_filepath, errors)

        self.logger.info("Model errors were calculated successfully")

    def __generate_list_of_cells(self):
        """
        generates a list containing all gridcell tuples (row, column) for a given number of rows and columns

        :return: List of gridcell tuples row, column)
        """

        list_of_cells = list()
        for row in range(self.number_of_grid_rows):
            for column in range(self.number_of_grid_columns):
                list_of_cells.append((row, column))
        return list_of_cells


def predict_for_gridcell(gridcell, cur_cell_df, number_of_prediction_timesteps,
                         order, seasonal_order, trend, number_of_timesteps_per_model):
    # split into train and test data
    cur_cell_train = cur_cell_df[:-number_of_prediction_timesteps]
    cur_cell_test = cur_cell_df[-number_of_prediction_timesteps:]

    predictions = np.zeros(shape=(len(cur_cell_test),))

    if cur_cell_train.mean() > 0.00675:
        print(f'Creating SARIMAX model for {gridcell=}')

        # create SARIMAX model
        model = SARIMAX(
            cur_cell_train,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        # fit the SARIMAX model
        model_fit = model.fit(disp=False)

        for step in range(number_of_prediction_timesteps):
            start = -number_of_prediction_timesteps + step - number_of_timesteps_per_model
            end = -number_of_prediction_timesteps + step
            # create new SARIMAX model
            current_sarima_model = SARIMAX(
                cur_cell_df[start:end],
                order=order,
                seasonal_order=seasonal_order,
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            # apply trained parameters
            current_sarima_model = current_sarima_model.filter(model_fit.params)
            # predict for current timestep
            predictions[step] = current_sarima_model.forecast()[0]

    predictions_df = pd.DataFrame(predictions)
    predictions_df.columns = [gridcell]

    return predictions_df
