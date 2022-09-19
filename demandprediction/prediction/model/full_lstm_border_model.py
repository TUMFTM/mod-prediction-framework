import logging
from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from demandprediction.prediction.generators.lstm_border_generator import LSTMBorderDataHandler
from demandprediction.prediction.model import BaseModel
from demandprediction.prediction.postprocessing.error_measures import log_grid_error
from demandprediction.prediction.postprocessing.postprocessing import calculate_grid_error
from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor
from demandprediction.utils.plot_utils import tum_colors
from demandprediction.utils.shared_filenames import get_predictions_filename

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class LSTMBorderModel(BaseModel):
    def __init__(self, config):
        """
        Initializes the Full LSTM Border model.
        This model builds one single LSTM for each grid and border cell.
        :param config: Configuration for the prediction.
        """
        super().__init__(model_name=type(self).__name__, data_directory=config['data_directory'])
        # Prediction interval start time
        self.start_date = config['start_date']
        # Prediction interval stop time
        self.stop_date = config['stop_date']
        #
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
        self.border_width = config['border_width']
        prediction_relative_path = get_predictions_filename(self.model_name, self.start_time)
        self.prediction_filepath = self.data_directory.joinpath(prediction_relative_path)
        self.number_of_prediction_timesteps = 0
        self.list_of_gridcells = self.__generate_list_of_cells()
        grid_models: Dict[str, Union[None, tf.keras.models.Model]] = {
            gridcell: None for gridcell in self.list_of_gridcells
        }
        self.border_orientations = [
            'north',
            'east',
            'south',
            'west'
        ]
        border_models: Dict[str, Union[None, tf.keras.models.Model]] = {
            orientation: None for orientation in self.border_orientations
        }
        self.combined_models = {**grid_models, **border_models}
        self.grid_predictions = None
        self.border_predictions = None
        self.data_handler: Union[None, LSTMBorderDataHandler] = None

        self.showed_model_info_once = False

        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
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

        self.number_of_prediction_timesteps = len(
            DataPreprocessor.get_date_range_between_dates(
                start_date=self.start_date,
                end_date=self.stop_date,
                freq=f'{self.step_size}min',
                end_inclusive=False
            )
        )

        self.data_handler = LSTMBorderDataHandler(
            filepath=self.ground_truth_filepath,
            number_of_prediction_timesteps=self.number_of_prediction_timesteps,
            gridcells=self.list_of_gridcells,
            border_orientations=self.border_orientations,
            lags=self.lags,
            shift=1,
            batch_size=self.batch_size
        )
        # set flag
        self.data_prepared = True

    def generate_model(self):
        for model_identifier in self.combined_models:
            self._generate_lstm_model(model_identifier)

        self.model = True

    def _generate_lstm_model(self, model_identifier):
        self.logger.info(f"Generating model for {model_identifier=}")
        lstm_input = Input(shape=(self.lags, 1))
        lstm1 = LSTM(32, stateful=False, return_sequences=True)(lstm_input)
        lstm2 = LSTM(32)(lstm1)

        mlp_input = Input(shape=(39,))
        mlp_dense = Dense(64, activation='relu')(mlp_input)
        mlp_dense = Dense(32, activation='relu')(mlp_dense)

        conc = keras.layers.concatenate([lstm2, mlp_dense])

        dense = Dense(64, activation='relu')(conc)
        dense = Dense(64, activation='relu')(dense)
        dense = keras.layers.Dropout(0.4)(dense)
        output = Dense(1)(dense)

        model = Model(inputs=[lstm_input, mlp_input], outputs=[output])
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()],
            loss_weights=[1.]
        )

        self.combined_models[model_identifier] = model

        if not self.showed_model_info_once:
            self.showed_model_info_once = True
            model.summary(print_fn=self.logger.info)
            filename = self.data_directory.joinpath(
                "visualization/{}_{}_{date:%Y-%m-%d_%H-%M-%S}.png".format(self.model_name, model_identifier,
                                                                          date=self.start_time)
            )
            plot_model(model, to_file=str(filename.resolve()),
                       show_shapes=True, show_layer_names=True)

    def train_model(self, plot_fitting_history=None, save_model=True):
        super().train_model()

        for model_identifier, model in self.combined_models.items():
            self.logger.info(f"Training model for {model_identifier=}")
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=15, verbose=2, mode='auto',
                restore_best_weights=True
            )

            history = model.fit(x=self.data_handler.get_train_ds(model_identifier),
                                epochs=self.epochs,
                                verbose=2,
                                validation_data=self.data_handler.get_val_ds(model_identifier),
                                shuffle=True,
                                callbacks=[early_stop])

            if plot_fitting_history is True:
                self._plot_training(history, model_identifier)
            if save_model:
                self._save_model(model, model_identifier)

        self.trained = True

    def _plot_training(self, history, model_name):
        plt.figure()
        plt.plot(history.history['loss'], color=tum_colors[2][1])
        plt.plot(history.history['val_loss'], color=tum_colors[0][0])
        plt.title('Training error')
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        filename = str(
            self.data_directory.joinpath(
                "visualization/{}_{}_training_{date:%Y-%m-%d_%H-%M-%S}.pdf".format(
                    self.model_name, model_name,
                    date=self.start_time
                )
            ).resolve()
        )
        plt.savefig(filename, bbox_inches='tight', dpi=300)

    def _save_model(self, model, model_name):
        model.save(
            str(
                self.data_directory.joinpath(
                    "models/{}_{}_{date:%Y-%m-%d_%H-%M-%S}.h5".format(
                        self.model_name, model_name,
                        date=self.start_time
                    )
                ).resolve()
            )
        )

    def predict(self):
        super().predict()
        # make predictions
        grid_predictions = {}
        for gridcell in self.list_of_gridcells:
            model = self.combined_models[gridcell]
            cell_prediction = model.predict(self.data_handler.get_test_ds(gridcell))
            cell_prediction = self.data_handler.get_scaler(gridcell).inverse_transform(cell_prediction)
            cell_prediction = cell_prediction.reshape(cell_prediction.shape[0])
            grid_predictions[gridcell] = cell_prediction

        border_predictions = {}
        for orientation in self.border_orientations:
            model = self.combined_models[orientation]
            predictions = model.predict(self.data_handler.get_test_ds(orientation))
            predictions = self.data_handler.get_scaler(orientation).inverse_transform(predictions)
            predictions = predictions.reshape(predictions.shape[0])
            border_predictions[orientation] = predictions

        self.grid_predictions = grid_predictions
        self.border_predictions = border_predictions
        self.predictions = True
        self.logger.info('Prediction successful')

    def save_predictions(self):
        super().save_predictions()
        grid_predictions_df = pd.DataFrame(self.grid_predictions)

        border_predictions_df = pd.DataFrame(self.border_predictions)

        # load actual data for index
        # noinspection PyTypeChecker
        actual_data_df = pd.read_hdf(self.ground_truth_filepath, key='even_grid')  # type: pd.DataFrame
        grid_predictions_df.index = actual_data_df.index[-len(grid_predictions_df.index):]
        border_predictions_df.index = actual_data_df.index[-len(border_predictions_df.index):]

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
            'nb_filters': [self.number_of_filters],
            'k_size': [self.kernel_size],
            'batch_size': [self.batch_size],
            'lr': [self.learning_rate],
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
