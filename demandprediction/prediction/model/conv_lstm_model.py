import logging
import os

import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, ConvLSTM2D, Dense, Reshape, add
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from demandprediction.prediction.generators.conv_lstm_add_info import ConvLSTMAdditionalInfoGenerator
from demandprediction.prediction.layers.hadamard import Hadamard
from demandprediction.prediction.model import ConvLSTMBaseModel
from demandprediction.utils.plot_utils import tum_colors


class ConvLSTMModel(ConvLSTMBaseModel):

    def __init__(self, config: dict):
        """
        Initializes the ConvLSTM model.
        Inputs are: Demand and Metainformation
        The Architecture does use a Hadamard layer
        :param config: Configuration for the prediction.
        """
        super().__init__(model_name=type(self).__name__, config=config)

        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        """
        Generates grid timeseries for whole area in the desired resolution.
        """
        super().prepare_data()

    def generate_model(self):
        # input for ConvLSTM
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
            filters=2 * self.number_of_filters,
            kernel_size=(2 * self.kernel_size, 2 * self.kernel_size),
            data_format='channels_last',
            padding='same',
            name='ConvLSTM2'
        )(convlstm1)

        convlstm_dense = Dense(100, activation='relu', name='FC1')(convlstm2)
        convlstm_dense = Dense(100, activation='relu', name='FC2')(convlstm_dense)
        convlstm_dense = Dense(100, activation='relu', name='FC3')(convlstm_dense)
        convlstm_dense = Dense(1, activation='relu', name='Output_ConvLSTM')(convlstm_dense)
        convlstm_dense_add = Hadamard(name='Hadamard1')(convlstm_dense)

        mlp_input = Input(shape=(37,), name='Metainformationen')
        mlp_dense = Dense(250, activation='relu', name='FC4')(mlp_input)
        mlp_dense = Dense(250, activation='relu', name='FC5')(mlp_dense)
        mlp_dense = Dense(self.number_of_grid_rows * self.number_of_grid_columns,
                          activation='relu', name='FC6')(mlp_dense)
        mlp_dense_output = Reshape(
            (self.number_of_grid_rows, self.number_of_grid_columns, 1),
            name='Output_Info'
        )(mlp_dense)
        mlp_dense_output = Hadamard(name='Hadamard2')(mlp_dense_output)

        main_output = add([convlstm_dense_add, mlp_dense_output], name='Output_Gesamt')

        # compile the model
        model = Model(inputs=[convlstm_input, mlp_input], outputs=[main_output])
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error',
                      loss_weights=[1.])

        filename = os.path.join(
            self.data_directory,
            "visualization/{}_{date:%Y-%m-%d_%H-%M-%S}.png".format(self.model_name,
                                                                   date=self.start_time)
        )
        plot_model(model, to_file=filename,
                   show_shapes=True, show_layer_names=True)

        self.model = model
        super().generate_model()
        model.summary(print_fn=self.logger.info)

    def train_model(self, plot_fitting_history=None, save_model=True):
        """
        Trains the generated model.
        :param plot_fitting_history: If set to True plots the training history for neural networks.
        :param save_model: If set to True, saves the model.
        """

        super().train_model()
        # create train and validation generator
        train_generator = ConvLSTMAdditionalInfoGenerator(
            filepath=self.ground_truth_filepath,
            train_val_test="train",
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            lags=self.lags,
            batch_size=self.batch_size,
            number_of_prediction_timesteps=self.number_of_prediction_timesteps,
            only_input=False
        )
        val_generator = ConvLSTMAdditionalInfoGenerator(
            filepath=self.ground_truth_filepath,
            train_val_test="val",
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            lags=self.lags,
            batch_size=self.batch_size,
            number_of_prediction_timesteps=self.number_of_prediction_timesteps,
            only_input=False
        )

        # early stopping to avoid overfitting
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=0, patience=3, verbose=2, mode='auto', restore_best_weights=True
        )
        # start training
        history = self.model.fit(x=train_generator,
                                 epochs=self.epochs,
                                 verbose=1,
                                 validation_data=val_generator,
                                 use_multiprocessing=False,
                                 shuffle=False,
                                 callbacks=[early_stop])

        # Possibility to plot training history
        if plot_fitting_history is True:
            plt.figure()
            plt.plot(history.history['loss'], color=tum_colors[2][1])
            plt.plot(history.history['val_loss'], color=tum_colors[0][0])
            plt.title('Training error')
            plt.ylabel('Error')
            plt.xlabel('Epoch')
            plt.legend(['Training', 'Validation'], loc='upper right')
            filename = os.path.join(
                self.data_directory,
                "visualization/{}_training_{date:%Y-%m-%d_%H-%M-%S}.pdf".format(
                    self.model_name,
                    date=self.start_time
                )
            )
            plt.savefig(filename, bbox_inches='tight', dpi=300)

        # Possibility to save the trained model
        if save_model:
            self.model.save(os.path.join(
                self.data_directory,
                "models/{}_{date:%Y-%m-%d_%H-%M-%S}.h5".format(
                    self.model_name,
                    date=self.start_time
                )
            ))

        self.trained = True
        self.logger.info("Model trained successfully")

    def predict(self):
        super().predict()
        # create test generator
        test_generator = ConvLSTMAdditionalInfoGenerator(
            filepath=self.ground_truth_filepath,
            train_val_test="test",
            number_of_rows=self.number_of_grid_rows,
            number_of_columns=self.number_of_grid_columns,
            lags=self.lags,
            batch_size=1,
            number_of_prediction_timesteps=self.number_of_prediction_timesteps,
            only_input=True
        )
        # make predictions
        predictions = self.model.predict(test_generator, use_multiprocessing=False, verbose=1)
        predictions = predictions.reshape(predictions.shape[0],
                                          self.number_of_grid_rows, self.number_of_grid_columns)
        self.predictions = predictions
        self.logger.info('Prediction successful')

    def save_predictions(self):
        super().save_predictions()

    def score_model(self):
        super().score_model()
