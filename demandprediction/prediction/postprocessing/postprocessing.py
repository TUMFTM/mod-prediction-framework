import numpy as np
import pandas as pd

from .error_measures import (
    calculate_average_rmse_of_cells_above_ceiling,
    calculate_average_mae_of_cells_above_ceiling,
    calculate_average_mapec_of_cells_above_ceiling,
    calculate_rmse,
    calculate_mae,
    calculate_mape,
    calculate_mapec
)


def calculate_grid_error(filepath, prediction_filepath, number_of_rows, number_of_columns, ceiling,
                         logger, actual_df_key='even_grid', prediction_df_key='predictions'):
    """
    Calculate all evaluation metrics for an Info model
    :param filepath: Filepath of the ground truth
    :param prediction_filepath: Filepath of the predictions
    :param number_of_rows: number of rows of the grid
    :param number_of_columns: number of columns of the grid
    :param ceiling: Minimum average rides
    :param logger: Logger for outputting information
    :return: Six error measures (tuple)
    """
    # Load necessary data into DataFrames
    actual_data_df = pd.read_hdf(filepath, key=actual_df_key)
    prediction_df = pd.read_hdf(prediction_filepath, key=prediction_df_key)
    # Fill possible NaN with zero
    actual_data_df = actual_data_df.fillna(0)
    prediction_df = prediction_df.fillna(0)
    # get numpy arrays out of DataFrames and reshape them
    prediction = prediction_df.values
    prediction = prediction.reshape(prediction.shape[0], number_of_rows, number_of_columns)
    train_data = actual_data_df[:-prediction.shape[0]].values
    actual_data = actual_data_df[-prediction.shape[0]:].values

    train_data = train_data.reshape(train_data.shape[0], number_of_rows, number_of_columns)
    actual_data = actual_data.reshape(actual_data.shape[0], number_of_rows, number_of_columns)
    logger.info(f'actual_data.shape: {actual_data.shape}')
    logger.info(f'train_data.shape: {train_data.shape}')
    logger.info(f'prediction_data.shape: {prediction.shape}')
    # Calculate error values for focus predictions
    avg_grid_rmse_above_ceiling = np.round(calculate_average_rmse_of_cells_above_ceiling(
        actual_data, prediction, train_data, ceiling), 3)
    avg_grid_mae_above_ceiling = np.round(calculate_average_mae_of_cells_above_ceiling(
        actual_data, prediction, train_data, ceiling), 3)
    avg_grid_mapec_above_ceiling = np.round(calculate_average_mapec_of_cells_above_ceiling(
        actual_data, prediction, train_data, ceiling), 3)
    global_rmse = np.round(
        calculate_rmse(
            np.sum(actual_data, axis=(1, 2)),
            np.sum(prediction, axis=(1, 2))
        ),
        decimals=3
    )
    global_mae = np.round(
        calculate_mae(
            np.sum(actual_data, axis=(1, 2)),
            np.sum(prediction, axis=(1, 2))
        ),
        decimals=3
    )
    global_mape = np.round(
        calculate_mapec(
            np.sum(actual_data, axis=(1, 2)),
            np.sum(prediction, axis=(1, 2))
        ),
        decimals=3
    )
    return (
        avg_grid_rmse_above_ceiling,
        avg_grid_mae_above_ceiling,
        avg_grid_mapec_above_ceiling,
        global_rmse,
        global_mae,
        global_mape
    )


def calculate_multistep_grid_error(filepath, prediction_filepath, number_of_rows, number_of_columns, ceiling,
                                   logger, actual_df_key='even_grid', prediction_horizon=1,
                                   prediction_df_key='predictions'):
    """
    Calculate all evaluation metrics for an Info model
    :param filepath: Filepath of the ground truth
    :param prediction_filepath: Filepath of the predictions
    :param number_of_rows: number of rows of the grid
    :param number_of_columns: number of columns of the grid
    :param ceiling: Minimum average rides
    :param prediction_horizon: Number of prediction time steps
    :param logger: Logger for outputting information
    :return: Six error measures (tuple) for every prediction horizon t_n as dict
    """
    # Load necessary data into DataFrames

    errors={}
    for t in range(prediction_horizon):

        actual_data_df = pd.read_hdf(filepath, key=actual_df_key)
        prediction_df = pd.read_hdf(prediction_filepath, key=prediction_df_key)

        prediction_df = get_predictions_for_step(prediction_df, t)
        # Fill possible NaN with zero
        actual_data_df = actual_data_df.fillna(0)
        prediction_df = prediction_df.fillna(0)
        # get numpy arrays out of DataFrames and reshape them
        prediction = prediction_df.values
        prediction = prediction.reshape(prediction.shape[0], number_of_rows, number_of_columns)
        train_data = actual_data_df[:-prediction.shape[0]].values
        actual_data = actual_data_df[-prediction.shape[0]:].values

        train_data = train_data.reshape(train_data.shape[0], number_of_rows, number_of_columns)
        actual_data = actual_data.reshape(actual_data.shape[0], number_of_rows, number_of_columns)
        logger.info(f'actual_data.shape: {actual_data.shape}')
        logger.info(f'train_data.shape: {train_data.shape}')
        logger.info(f'prediction_data.shape: {prediction.shape}')
        # Calculate error values for focus predictions
        avg_grid_rmse_above_ceiling = np.round(calculate_average_rmse_of_cells_above_ceiling(
            actual_data, prediction, train_data, ceiling), 3)
        avg_grid_mae_above_ceiling = np.round(calculate_average_mae_of_cells_above_ceiling(
            actual_data, prediction, train_data, ceiling), 3)
        avg_grid_mapec_above_ceiling = np.round(calculate_average_mapec_of_cells_above_ceiling(
            actual_data, prediction, train_data, ceiling), 3)
        global_rmse = np.round(
            calculate_rmse(
                np.sum(actual_data, axis=(1, 2)),
                np.sum(prediction, axis=(1, 2))
            ),
            decimals=3
        )
        global_mae = np.round(
            calculate_mae(
                np.sum(actual_data, axis=(1, 2)),
                np.sum(prediction, axis=(1, 2))
            ),
            decimals=3
        )
        global_mape = np.round(
            calculate_mapec(
                np.sum(actual_data, axis=(1, 2)),
                np.sum(prediction, axis=(1, 2))
            ),
            decimals=3
        )
        errors[f't_{t}'] = (avg_grid_rmse_above_ceiling,
            avg_grid_mae_above_ceiling,
            avg_grid_mapec_above_ceiling,
            global_rmse,
            global_mae,
            global_mape)

    return errors


def calculate_citywide_error(filepath, prediction_filepath, prediction_horizon,
                             logger, actual_df_key='citywide', prediction_df_key='predictions'):
    # Load necessary data into DataFrames
    actual_data_df = pd.read_hdf(filepath, key=actual_df_key)
    prediction_df = pd.read_hdf(prediction_filepath, key=prediction_df_key)
    # Fill possible NaN with zero
    actual_data_df = actual_data_df.fillna(0)
    prediction_df = prediction_df.fillna(0)
    # get numpy arrays out of DataFrames and reshape them
    prediction = prediction_df.values
    prediction = prediction.reshape(prediction.shape[0], prediction_horizon)
    actual_data = np.zeros(shape=(prediction.shape[0], prediction_horizon))
    for timestep in range(actual_data.shape[0]):
        start = -actual_data.shape[0] - prediction_horizon + 1 + timestep
        end = start + prediction_horizon if start < -prediction_horizon else None
        actual_data[timestep] = actual_data_df[start:end].values.reshape(prediction_horizon)

    rmse_list = []
    mae_list = []
    mape_list = []
    for timestep in range(prediction_horizon):
        rmse_list.append(
            calculate_rmse(actual_data[:, timestep], prediction[:, timestep])
        )
        mae_list.append(
            calculate_mae(actual_data[:, timestep], prediction[:, timestep])
        )
        mape_list.append(
            calculate_mape(actual_data[:, timestep], prediction[:, timestep])
        )
        logger.info(f"Errors for {timestep=}:"
                    f" RMSE: {rmse_list[timestep]},"
                    f" MAE: {mae_list[timestep]},"
                    f" MAPE: {mape_list[timestep]}")

    global_rmse = np.round(
        np.mean(rmse_list),
        decimals=3
    )
    global_mae = np.round(
        np.mean(mae_list),
        decimals=3
    )
    global_mape = np.round(
        np.mean(mape_list),
        decimals=3
    )
    return (
        global_rmse,
        global_mae,
        global_mape
    )

def get_predictions_for_step(multi_step_predictions:pd.DataFrame, step=0):
    predictions = multi_step_predictions.applymap(lambda x :x[step])
    predictions.index = predictions.index.shift(step)
    return predictions