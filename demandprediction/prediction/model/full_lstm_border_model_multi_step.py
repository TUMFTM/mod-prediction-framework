import logging
from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from demandprediction.prediction.generators.lstm_border_generator_multi_step import LSTMBorderDataHandlerMultiStep
from demandprediction.prediction.model import BaseModel
from demandprediction.prediction.postprocessing.error_measures import log_grid_error
from demandprediction.prediction.postprocessing.postprocessing import calculate_multistep_grid_error
from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor
from demandprediction.utils.plot_utils import tum_colors
from demandprediction.utils.shared_filenames import get_predictions_filename

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class LSTMBorderModelMultiStep(BaseModel):
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
        self.prediction_horizon = config['prediction_horizon']

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
        self.data_handler: Union[None, LSTMBorderDataHandlerMultiStep] = None

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

        self.data_handler = LSTMBorderDataHandlerMultiStep(
            filepath=self.ground_truth_filepath,
            number_of_prediction_timesteps=self.number_of_prediction_timesteps,
            gridcells=self.list_of_gridcells,
            border_orientations=self.border_orientations,
            lags=self.lags,
            prediction_horizon = self.prediction_horizon,
            shift=self.prediction_horizon,
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

        # 39 is derived from hour, weekday, holiday, oktoberfest, school_breaks
        mlp_input_size = 39 #+self.prediction_horizon if self.prediction_horizon > 1 else 39
        mlp_input = Input(shape=(mlp_input_size,), name= 'Metainformation')
        mlp_dense = Dense(64, activation='relu')(mlp_input)
        mlp_dense = Dense(32, activation='relu')(mlp_dense)

        conc = keras.layers.concatenate([lstm2, mlp_dense], name = 'Concatenate')

        dense = Dense(64, activation='relu', name ='FC3')(conc)
        dense = Dense(64, activation='relu', name = 'FC4')(dense)
        dense = keras.layers.Dropout(0.4)(dense)
        output = Dense(self.prediction_horizon, name = 'Output')(dense)
        output = keras.layers.Reshape([self.prediction_horizon, 1])(output)

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
            cell_prediction = cell_prediction.reshape(cell_prediction.shape[0], self.prediction_horizon)
            grid_predictions[gridcell] = cell_prediction.tolist()

        border_predictions = {}
        for orientation in self.border_orientations:
            model = self.combined_models[orientation]
            predictions = model.predict(self.data_handler.get_test_ds(orientation))
            predictions = self.data_handler.get_scaler(orientation).inverse_transform(predictions)
            predictions = predictions.reshape(predictions.shape[0], self.prediction_horizon)
            border_predictions[orientation] = predictions.tolist()

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
        grid_predictions_df.index = actual_data_df.index[
                               -len(grid_predictions_df.index) -
                               self.data_handler.offset_from_end:
                               - self.data_handler.offset_from_end]
        border_predictions_df.index = actual_data_df.index[
                               -len(border_predictions_df.index) -
                               self.data_handler.offset_from_end:
                               - self.data_handler.offset_from_end]

        grid_predictions_df.to_hdf(self.prediction_filepath, key='grid_predictions', complevel=6)
        border_predictions_df.to_hdf(self.prediction_filepath, key='border_predictions', complevel=6)

    def score_model(self):
        super().score_model()

        # Calculate error
        model_id = "{}_{date:%Y-%m-%d_%H-%M-%S}.h5".format(
            self.model_name, date=self.start_time)

        errors = calculate_multistep_grid_error(
            filepath=self.ground_truth_filepath,
            prediction_filepath=self.prediction_filepath,
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            prediction_horizon=self.prediction_horizon,
            ceiling=1,
            logger=self.logger,
            actual_df_key='even_grid',
            prediction_df_key='grid_predictions'
        )

        # Write the errors and all hyperparams to a csv file!
        df = pd.DataFrame({
            'model_id': [model_id],
            'model_name': [self.model_name],
            'number_of_rows': [self.number_of_grid_rows],
            'number_of_columns': [self.number_of_grid_columns],
            'lags': [self.lags],
            'nb_filters': [self.number_of_filters],
            'k_size': [self.kernel_size],
            'batch_size': [self.batch_size],
            'lr': [self.learning_rate],
            'step_size': [self.step_size],
            'prediction_horizon' :[self.prediction_horizon]
        })
        for i in range(self.prediction_horizon):
            df[f't_{i}_avg_grid_rmse_above_ceiling'] = errors[f't_{i}'][0]
            df[f't_{i}_avg_grid_mae_above_ceiling'] = errors[f't_{i}'][1]
            df[f't_{i}_avg_grid_mapec_above_ceiling']=  errors[f't_{i}'][2]
            df[f't_{i}_global_rmse'] = errors[f't_{i}'][3]
            df[f't_{i}_global_mae'] = errors[f't_{i}'][4]
            df[f't_{i}_global_mape'] = errors[f't_{i}'][5]

            log_grid_error(self.logger, self.prediction_filepath, errors[f't_{i}'])

        filename = self.data_directory.joinpath("evaluation/metrics_and_hyperparams.csv")
        with open(filename, 'a') as f:
            df.to_csv(filename, mode='a', header=not f.tell())



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
