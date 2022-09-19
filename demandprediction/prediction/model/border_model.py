import logging
from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, ConvLSTM2D, Dense, Reshape, add, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from demandprediction.prediction.generators.border_generator import BorderDataHandler
from demandprediction.prediction.layers.hadamard import Hadamard
from demandprediction.prediction.model import BaseModel
from demandprediction.prediction.postprocessing.error_measures import log_grid_error
from demandprediction.prediction.postprocessing.postprocessing import calculate_grid_error
from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor
from demandprediction.utils.plot_utils import tum_colors
from demandprediction.utils.shared_filenames import get_predictions_filename

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class BorderModel(BaseModel):

    def __init__(self, config):
        """
        Initializes the Border model.
        :param config: Configuration for the prediction.
        """
        super().__init__(model_name=type(self).__name__, data_directory=config['data_directory'])
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

        # Calculate number of prediction timesteps
        self.number_of_prediction_timesteps = len(
            DataPreprocessor.get_date_range_between_dates(
                start_date=self.start_date,
                end_date=self.stop_date,
                freq=f'{self.step_size}min',
                end_inclusive=False
            )
        )

        # Create Data handler
        self.data_handler = BorderDataHandler(
            filepath=self.ground_truth_filepath,
            number_of_prediction_timesteps=self.number_of_prediction_timesteps,
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            lags=self.lags,
            shift=1,
            batch_size=self.batch_size
        )
        # set flag
        self.data_prepared = True

    def generate_model(self):
        self._generate_grid_model()
        for orientation in self.border_models:
            self._generate_border_model(orientation)

    def _generate_grid_model(self):
        self.logger.info("Generating model for grid")

        convlstm_input = Input(shape=(self.lags, self.number_of_grid_rows, self.number_of_grid_columns, 1),
                               name='Fahrten_pro_Zelle')

        # first ConvLSTM-Layer
        convlstm1 = ConvLSTM2D(
            filters=self.number_of_filters,
            kernel_size=(self.kernel_size, self.kernel_size),
            data_format='channels_last',
            padding='same',
            return_sequences=True,
            name='ConvLSTM1'
        )(convlstm_input)

        # second ConvLSTM-Layer with doubled number of filters and doubled kernel_size
        convlstm2 = ConvLSTM2D(
            filters=self.number_of_filters,
            kernel_size=(self.kernel_size, self.kernel_size),
            data_format='channels_last',
            padding='same',
            name='ConvLSTM2'
        )(convlstm1)

        convlstm_dense = Dense(50, activation='relu', name='FC1')(convlstm2)
        convlstm_dense = Dense(50, activation='relu', name='FC3')(convlstm_dense)
        convlstm_dense = Dense(1, activation='relu', name='Output_ConvLSTM')(convlstm_dense)
        convlstm_dense_add = Hadamard(name='Hadamard1')(convlstm_dense)

        mlp_input = Input(shape=(39,), name='Metainformationen')
        mlp_dense = Dense(50, activation='relu', name='FC4')(mlp_input)
        mlp_dense = Dense(50, activation='relu', name='FC5')(mlp_dense)
        mlp_dense = Dense(self.number_of_grid_rows * self.number_of_grid_columns,
                          activation='relu', name='FC6')(mlp_dense)
        mlp_dense_output = Reshape(
            (self.number_of_grid_rows, self.number_of_grid_columns, 1),
            name='Output_Info'
        )(mlp_dense)
        mlp_dense_output = Hadamard(name='Hadamard2')(mlp_dense_output)

        main_output = add([convlstm_dense_add, mlp_dense_output], name='Output_Gesamt')

        model = Model(inputs=[convlstm_input, mlp_input], outputs=[main_output])

        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()],
            loss_weights=[1.]
        )

        self.model = model
        model.summary(print_fn=self.logger.info)

        filename = self.data_directory.joinpath(
            "visualization/{}_{}_{date:%Y-%m-%d_%H-%M-%S}.png".format(self.model_name, 'grid',
                                                                      date=self.start_time)
        )
        plot_model(model, to_file=str(filename.resolve()),
                   show_shapes=True, show_layer_names=True)

    def _generate_border_model(self, orientation):
        self.logger.info(f"Generation model for {orientation} border")
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

        border_model = Model(inputs=[lstm_input, mlp_input], outputs=[output])
        border_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()],
            loss_weights=[1.]
        )

        border_model.summary(print_fn=self.logger.info)
        self.border_models[orientation] = border_model

        filename = self.data_directory.joinpath(
            "visualization/{}_{}_{date:%Y-%m-%d_%H-%M-%S}.png".format(self.model_name, orientation,
                                                                      date=self.start_time)
        )
        plot_model(border_model, to_file=str(filename.resolve()),
                   show_shapes=True, show_layer_names=True)

    def train_model(self, plot_fitting_history=None, save_model=True):
        super().train_model()
        self.logger.info("Training grid model")

        # early stopping to avoid overfitting
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=4, verbose=2, mode='auto', restore_best_weights=True
        )
        log_dir = self.data_directory.joinpath(
            'logs/fit/{}_training_{date:%Y-%m-%d_%H-%M-%S}'.format(
                self.model_name,
                date=self.start_time
            )
        )
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # start training
        history = self.model.fit(x=self.data_handler.train_grid,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=self.data_handler.val_grid,
                                 shuffle=True,
                                 callbacks=[early_stop, tensorboard_callback])

        # Possibility to plot training history
        if plot_fitting_history is True:
            self._plot_training(history, 'grid')
        # Possibility to save the trained model
        if save_model:
            self._save_model(self.model, 'grid')

        for orientation, border_model in self.border_models.items():
            self.logger.info(f"Training {orientation} border model")
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0, patience=20, verbose=2, mode='auto',
                restore_best_weights=True
            )
            log_dir = self.data_directory.joinpath(
                'logs/fit/{}_{}_training_{date:%Y-%m-%d_%H-%M-%S}'.format(
                    self.model_name, orientation,
                    date=self.start_time
                )
            )
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

            history = border_model.fit(x=self.data_handler.get_border_train(orientation),
                                       epochs=self.epochs,
                                       verbose=1,
                                       validation_data=self.data_handler.get_border_val(orientation),
                                       shuffle=True,
                                       callbacks=[early_stop, tensorboard_callback])
            if plot_fitting_history is True:
                self._plot_training(history, orientation)
            if save_model:
                self._save_model(border_model, orientation)

        self.trained = True
        self.logger.info("Model trained successfully")

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
        grid_predictions = self.model.predict(self.data_handler.test_grid, verbose=1)
        grid_predictions = grid_predictions.reshape((grid_predictions.shape[0],
                                                     self.number_of_grid_rows * self.number_of_grid_columns))
        grid_predictions = self.data_handler.grid_scaler.inverse_transform(grid_predictions)
        grid_predictions = grid_predictions.reshape((grid_predictions.shape[0],
                                                     self.number_of_grid_rows, self.number_of_grid_columns))
        self.grid_predictions = grid_predictions

        border_predictions = {}
        for orientation, border_model in self.border_models.items():
            predictions = border_model.predict(self.data_handler.get_border_test(orientation), verbose=1)
            predictions = self.data_handler.border_data[orientation].scaler.inverse_transform(predictions)
            predictions = predictions.reshape(predictions.shape[0])
            border_predictions[orientation] = predictions

        self.border_predictions = border_predictions
        self.predictions = True
        self.logger.info('Prediction successful')

    def save_predictions(self):
        super().save_predictions()
        grid_predictions = self.grid_predictions.reshape(self.grid_predictions.shape[0],
                                                         self.number_of_grid_rows *
                                                         self.number_of_grid_columns)
        grid_predictions_df = pd.DataFrame(grid_predictions)

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
