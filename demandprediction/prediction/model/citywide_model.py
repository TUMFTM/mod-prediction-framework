import logging

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from demandprediction.prediction.generators.citywide_generator import CitywideDataHandler
from demandprediction.prediction.model import BaseModel
from demandprediction.prediction.postprocessing.error_measures import log_citywide_error
from demandprediction.prediction.postprocessing.postprocessing import calculate_citywide_error
from demandprediction.prediction.preprocessing.preprocessing import DataPreprocessor
from demandprediction.utils.plot_utils import tum_colors
from demandprediction.utils.shared_filenames import get_predictions_filename

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class CitywideModel(BaseModel):

    def __init__(self, config):
        """
        Initializes the Border model.
        :param config: Configuration for the prediction.
        """
        super().__init__(model_name=type(self).__name__, data_directory=config['data_directory'])
        self.start_date = config['start_date']
        self.stop_date = config['stop_date']
        self.lags = config['lags']
        self.number_of_filters = config['number_of_filters']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.step_size = config['step_size']
        self.epochs = config['epochs']
        self.prediction_horizon = config['prediction_horizon']
        prediction_relative_path = get_predictions_filename(self.model_name, self.start_time)
        self.prediction_filepath = self.data_directory.joinpath(prediction_relative_path)
        self.number_of_prediction_timesteps = 0
        self.data_handler = None

        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        super().prepare_data()
        data_preprocessor = DataPreprocessor(self.data_directory, None, None)
        self.ground_truth_filepath = data_preprocessor.generate_citywide_timeseries(
            step_size=self.step_size,
            stop_date=self.stop_date
        )

        self.number_of_prediction_timesteps = len(
            DataPreprocessor.get_date_range_between_dates(
                start_date=self.start_date,
                end_date=self.stop_date,
                freq=f'{self.step_size}min',
                end_inclusive=False
            )
        )

        self.data_handler = CitywideDataHandler(
            filepath=self.ground_truth_filepath,
            number_of_prediction_timesteps=self.number_of_prediction_timesteps,
            prediction_horizon=self.prediction_horizon,
            input_width=self.lags,
            shift=self.prediction_horizon,
            batch_size=self.batch_size
        )
        # set flag
        self.data_prepared = True

    def generate_model(self):
        lstm_input = Input(shape=(self.lags, 1), name='Gesamtanzahl_Fahrten')
        lstm1 = LSTM(32, stateful=False, return_sequences=True)(lstm_input)
        lstm2 = LSTM(32)(lstm1)

        # 27 is derived from months, weekday, holidays, and school_breaks
        mlp_input_size = 27 + self.prediction_horizon if self.prediction_horizon > 1 else 27
        mlp_input = Input(shape=(mlp_input_size,), name='Metainformationen')
        dense = Dense(64, activation='relu', name='FC1')(mlp_input)
        dense = Dense(32, activation='relu', name='FC2')(dense)

        conc = keras.layers.concatenate([lstm2, dense], name='Concatenate')

        dense = Dense(64, activation='relu', name='FC3')(conc)
        dense = keras.layers.Dropout(0.4)(dense)
        dense = Dense(64, activation='relu', name='FC4')(dense)
        dense = keras.layers.Dropout(0.4)(dense)
        dense = Dense(self.prediction_horizon, name='Output')(dense)
        dense = keras.layers.Reshape([self.prediction_horizon, 1])(dense)

        model = Model(inputs=[lstm_input, mlp_input], outputs=[dense])
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.losses.MeanSquaredError(),
            metrics=[tf.metrics.MeanAbsoluteError()],
            loss_weights=[1.]
        )

        filename = self.data_directory.joinpath(
            "visualization/{}_{date:%Y-%m-%d_%H-%M-%S}.png".format(self.model_name,
                                                                   date=self.start_time)
        )
        plot_model(model, to_file=str(filename.resolve()),
                   show_shapes=True, show_layer_names=True)

        self.model = model
        super().generate_model()
        model.summary(print_fn=self.logger.info)

    def train_model(self, plot_fitting_history=None, save_model=True):
        super().train_model()

        # early stopping to avoid overfitting
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=150, verbose=2, mode='auto', restore_best_weights=True
        )
        log_dir = self.data_directory.joinpath(
            'logs/fit/{}_training_{date:%Y-%m-%d_%H-%M-%S}'.format(
                self.model_name,
                date=self.start_time
            )
        )
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        # start training
        history = self.model.fit(x=self.data_handler.train,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=self.data_handler.val,
                                 shuffle=True,
                                 callbacks=[early_stop, tensorboard_callback])

        # Possibility to plot training history
        if plot_fitting_history is True:
            plt.figure()
            plt.plot(history.history['loss'], color=tum_colors[2][1])
            plt.plot(history.history['val_loss'], color=tum_colors[0][0])
            plt.title('Training error')
            plt.ylabel('Error')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='upper right')
            filename = str(
                self.data_directory.joinpath(
                    "visualization/{}_training_{date:%Y-%m-%d_%H-%M-%S}.pdf".format(
                        self.model_name,
                        date=self.start_time
                    )
                ).resolve()
            )
            plt.savefig(filename, bbox_inches='tight', dpi=300)

        # Possibility to save the trained model
        if save_model:
            self.model.save(
                str(
                    self.data_directory.joinpath(
                        "models/{}_{date:%Y-%m-%d_%H-%M-%S}.h5".format(
                            self.model_name,
                            date=self.start_time
                        )
                    ).resolve()
                )
            )

        self.trained = True
        self.logger.info("Model trained successfully")

    def predict(self):
        super().predict()
        # make predictions
        predictions = self.model.predict(self.data_handler.test, verbose=1)
        predictions = predictions.reshape(predictions.shape[0], self.prediction_horizon)
        predictions = self.data_handler.scaler.inverse_transform(predictions)
        self.logger.info(f"Predictions: {predictions[:2 * self.prediction_horizon, :]}")
        self.predictions = predictions
        self.logger.info('Prediction successful')

    def save_predictions(self):
        super().save_predictions()
        predictions_df = pd.DataFrame(self.predictions)

        # load actual data for index
        # noinspection PyTypeChecker
        actual_data_df = pd.read_hdf(self.ground_truth_filepath, key='citywide')  # type: pd.DataFrame
        predictions_df.index = actual_data_df.index[
                               -len(predictions_df.index) -
                               self.data_handler.offset_from_end:
                               - self.data_handler.offset_from_end
                               ]

        predictions_df.to_hdf(self.prediction_filepath, key='predictions', complevel=6)

    def score_model(self):
        # Calculate error
        model_id = "{}_{date:%Y-%m-%d_%H-%M-%S}.h5".format(
            self.model_name, date=self.start_time)

        errors = calculate_citywide_error(
            filepath=self.ground_truth_filepath,
            prediction_filepath=self.prediction_filepath,
            prediction_horizon=self.prediction_horizon,
            logger=self.logger
        )

        # Write the errors and all hyperparams to a csv file!
        df = pd.DataFrame({
            'model_id': [model_id],
            'model_name': [self.model_name],
            'avg_grid_rmse_above_ceiling': [""],
            'avg_grid_mae_above_ceiling': [""],
            'avg_grid_mapec_above_ceiling': [""],
            'global_rmse': [errors[0]],
            'global_mae': [errors[1]],
            'global_mape': [errors[2]],
            'number_of_rows': [""],
            'number_of_columns': [""],
            'lags': [self.lags],
            'nb_filters': [self.number_of_filters],
            'k_size': [""],
            'batch_size': [self.batch_size],
            'lr': [self.learning_rate],
            'step_size': [self.step_size],
        })

        filename = self.data_directory.joinpath("evaluation/metrics_and_hyperparams.csv")
        with open(filename, 'a') as f:
            df.to_csv(filename, mode='a', header=not f.tell())

        log_citywide_error(self.logger, self.prediction_filepath, errors)

        self.logger.info("Model errors were calculated successfully")
