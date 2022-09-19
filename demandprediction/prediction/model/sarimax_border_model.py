import concurrent.futures
import logging
from loguru import logger

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from demandprediction.prediction.model import BaseModel
from demandprediction.prediction.postprocessing.error_measures import log_grid_error
from demandprediction.prediction.postprocessing.postprocessing import calculate_grid_error
from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor
from demandprediction.utils.shared_filenames import get_predictions_filename


class SARIMAXBorderModel(BaseModel):
    def __init__(self, config: dict, order: tuple, seasonal_order: tuple, trend: str, max_workers: int):
        super().__init__(model_name=type(self).__name__, data_directory=config['data_directory'])
        self.start_date = config['start_date']
        self.stop_date = config['stop_date']
        self.number_of_grid_rows = config['number_of_grid_rows']
        self.number_of_grid_columns = config['number_of_grid_columns']
        self.step_size = config['step_size']
        self.x_limits = (config['x_min'], config['x_max'])
        self.y_limits = (config['y_min'], config['y_max'])
        self.border_width = config['border_width']
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.max_workers = max_workers

        self.border_orientations = [
            'north',
            'east',
            'south',
            'west'
        ]

        self.grid_predictions = None
        self.border_predictions = None

        self.grid_data = None

        self.number_of_timesteps_per_model = 7 * 24 * 3
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

        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)

    def prepare_data(self):
        """
        Generates the citywide timeseries and saves it to the disk.
        """
        super().prepare_data()
        data_preprocessor = DataPreprocessor(self.data_directory, self.x_limits, self.y_limits)
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

    def train_model(self, plot_fitting_history=None):
        self.logger.info("Model training isn´t needed. Just use predict.")

    # noinspection PyTypeChecker
    def predict(self):
        list_of_gridcells = self.__generate_list_of_cells()

        # load data once to minimize disk usage (bottleneck)
        self.grid_data = pd.read_hdf(self.ground_truth_filepath, key='even_grid')  # type: pd.DataFrame
        border_data = pd.read_hdf(self.ground_truth_filepath, key='border')  # type: pd.DataFrame

        list_of_grid_prediction_dfs = []
        list_of_border_prediction_dfs = []
        futures_to_grid_prediction = [
            self.executor.submit(predict_for_timeseries, gridcell, self.grid_data[gridcell],
                                 self.number_of_prediction_timesteps, self.order,
                                 self.seasonal_order, self.trend,
                                 self.number_of_timesteps_per_model)
            for gridcell in list_of_gridcells
        ]

        futures_to_border_predictions = [
            self.executor.submit(predict_for_timeseries, orientation, border_data[orientation],
                                 self.number_of_prediction_timesteps, self.order,
                                 self.seasonal_order, self.trend,
                                 self.number_of_timesteps_per_model)
            for orientation in self.border_orientations
        ]

        for future in concurrent.futures.as_completed(futures_to_grid_prediction):
            list_of_grid_prediction_dfs.append(future.result())

        for future in concurrent.futures.as_completed(futures_to_border_predictions):
            list_of_border_prediction_dfs.append(future.result())

        grid_predictions = pd.concat(list_of_grid_prediction_dfs, axis=1)
        grid_predictions = grid_predictions.reindex(sorted(grid_predictions.columns), axis=1)
        self.grid_predictions = grid_predictions

        border_predictions = pd.concat(list_of_border_prediction_dfs, axis=1)
        border_predictions = border_predictions.reindex(self.border_orientations, axis=1)
        self.border_predictions = border_predictions

        self.predictions = True
        self.logger.info('Prediction successful')

        self.executor.shutdown(wait=True)

    def save_predictions(self):
        super().save_predictions()

        # load actual data for index
        # noinspection PyTypeChecker
        self.grid_predictions.index = self.grid_data.index[-len(self.grid_predictions.index):]
        self.border_predictions.index = self.grid_data.index[-len(self.border_predictions.index):]

        self.grid_predictions.to_hdf(self.prediction_filepath, key='grid_predictions', complevel=6)
        self.border_predictions.to_hdf(self.prediction_filepath, key='border_predictions', complevel=6)

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
            'lags': [''],
            'nb_filters': [''],
            'k_size': [''],
            'batch_size': [''],
            'lr': [''],
            'step_size': [self.step_size],
        })

        filename = self.data_directory.joinpath("evaluation/metrics_and_hyperparams.csv")
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
                list_of_cells.append(str((row, column)))
        return list_of_cells


def predict_for_timeseries(name, current_df, number_of_prediction_timesteps,
                           order, seasonal_order, trend, number_of_timesteps_per_model):
    # split into train and test data
    cur_cell_train = current_df[:-number_of_prediction_timesteps]
    cur_cell_test = current_df[-number_of_prediction_timesteps:]

    predictions = np.zeros(shape=(len(cur_cell_test),))

    logger.info(f'Mean of train series {name=}: {cur_cell_train.mean()}')

    if cur_cell_train.mean() > 0.01:
        logger.info(f'Creating SARIMAX model for {name=}')

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
                current_df[start:end],
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
    else:
        logger.info(f"Skipping timeseries {name=} because of low demand ({cur_cell_train.mean()})")

    predictions_df = pd.DataFrame(predictions)
    predictions_df.columns = [name]

    return predictions_df
