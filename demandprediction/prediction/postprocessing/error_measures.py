from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_rmse_for_grid(pred_array, real_array):
    """
    Calculates the RMSE for a grid. The RMSE is calculated for each cell.
    :param pred_array: Predicted values
    :param real_array: True values
    :return: Array of the RMSE results
    """

    # create array for RMSE results
    rmse_results = np.zeros(shape=(pred_array.shape[1], pred_array.shape[2]))

    # calculate RMSE for each gridcell
    for row in range(pred_array.shape[1]):
        for col in range(pred_array.shape[2]):
            rmse_results[row, col] = calculate_rmse(real_array[:, row, col], pred_array[:, row, col])

    return rmse_results


def calculate_average_rmse_of_cells_above_ceiling(y_true, y_predicted, train_array, ceiling):
    """
    Calculates the average RMSE of cells that have more average rides than the ceiling
    :param y_true: True values
    :param y_predicted: Predicted values
    :param train_array: Array of the training data
    :param ceiling: Minimum average rides
    :return: average RMSE
    """
    train_array_mean = np.mean(train_array, axis=0)

    rmse_array = calculate_rmse_for_grid(y_predicted, y_true)

    res_array = np.where(train_array_mean < ceiling, 0, rmse_array)

    average_rmse = np.mean(res_array[np.nonzero(res_array)])

    return average_rmse


def calculate_rmse(actual, predicted):
    """
    Calculates the RMSE between the actual and predicted values.

    :param actual: Actual values.
    :param predicted: Predicted values.
    :return: RMSE
    """
    return sqrt(mean_squared_error(actual, predicted))


def calculate_mae_for_grid(pred_array, real_array):
    """
    Calculates the MAE for a grid. The MAE is calculated for each cell.
    :param pred_array: Predicted values
    :param real_array: True values
    :return: Array of the MAE results
    """

    # create array for  MAE results
    rmse_results = np.zeros(shape=(pred_array.shape[1], pred_array.shape[2]))

    # calculate RMSE for each gridcell
    for row in range(pred_array.shape[1]):
        for col in range(pred_array.shape[2]):
            rmse_results[row, col] = calculate_mae(real_array[:, row, col], pred_array[:, row, col])

    return rmse_results


def calculate_average_mae_of_cells_above_ceiling(y_true, y_predicted, train_array, ceiling):
    """
    Calculates the average MAE of cells that have more average rides than the ceiling
    :param y_true: True values
    :param y_predicted: Predicted values
    :param train_array: Array of the training data
    :param ceiling: Minimum average rides
    :return: average MAE
    """
    train_array_mean = np.mean(train_array, axis=0)

    rmse_array = calculate_mae_for_grid(y_predicted, y_true)

    res_array = np.where(train_array_mean < ceiling, 0, rmse_array)

    average_rmse = np.mean(res_array[np.nonzero(res_array)])

    return average_rmse


def calculate_mae(actual, predicted):
    """
    Calculates the RMSE between the actual and predicted values.

    :param actual: Actual values.
    :param predicted: Predicted values.
    :return: RMSE
    """
    return mean_absolute_error(actual, predicted)


def calculate_average_mapec_of_cells_above_ceiling(y_true, y_predicted, train_array, ceiling, offset=1):
    """
    Calculates the average MAPE_c of cells that have more average rides than the ceiling
    :param y_true: True values
    :param y_predicted: Predicted values
    :param train_array: Array of the training data
    :param ceiling: Minimum average rides
    :param offset: offset which is added to the denominator
    :return: average MAPE_c
    """
    train_array_mean = np.mean(train_array, axis=0)

    mapec_array = calculate_mapec_for_grid(y_predicted, y_true, offset=offset)

    res_array = np.where(train_array_mean < ceiling, 0, mapec_array)

    average_rmse = np.mean(res_array[np.nonzero(res_array)])

    return average_rmse


def calculate_mapec_for_grid(y_true, y_predicted, offset=1):
    """
    Calculates the MAPE_c for a grid. The MAPE_c is calculated for each cell.
    :param y_true: True values
    :param y_predicted: Predicted values
    :param offset: offset which is added to the denominator
    :return: Array of the MAPE_c results
    """
    # create array for the MAPE_c results
    rmse_results = np.zeros(shape=(y_predicted.shape[1], y_predicted.shape[2]))
    # calculate MAPE_c for each gridcell
    for row in range(y_predicted.shape[1]):
        for col in range(y_predicted.shape[2]):
            rmse_results[row, col] = calculate_mapec(y_true[:, row, col], y_predicted[:, row, col], offset)

    return rmse_results


def calculate_mapec(y_true, y_predicted, offset=1):
    """
    Calculate the MAPE_c of a timeseries.
    :param y_true: True values
    :param y_predicted: Predicted values
    :param offset: offset which is added to the denominator
    :return: MAPE_c
    """
    return np.mean(np.abs((y_true - y_predicted) / (y_true + offset))) * 100


def calculate_mape(actual, predicted):
    """
    Calculate the MAPE of a timeseries.
    :param actual: True values
    :param predicted: Predicted values
    :return: MAPE
    """
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def calculate_average_smape_of_cells_above_ceiling(y_true, y_predicted, train_array, ceiling, offset=1):
    """
    Calculates the average SMAPE of cells that have more average rides than the ceiling
    :param y_true: True values
    :param y_predicted: Predicted values
    :param train_array: Array of the training data
    :param ceiling: Minimum average rides
    :param offset: offset which is added to the denominator
    :return: average SMAPE
    """
    train_array_mean = np.mean(train_array, axis=0)

    smape_array = calculate_smape_for_grid(y_predicted, y_true, offset)

    res_array = np.where(train_array_mean < ceiling, 0, smape_array)

    average_smape = np.mean(res_array[np.nonzero(res_array)])

    return average_smape


def calculate_smape_for_grid(pred_array, real_array, offset=1):
    """
    Calculates the SMAPE for a grid. The SMAPE is calculated for each cell.
    :param pred_array: True values
    :param real_array: Prdicted values
    :param offset: offset which is added to the denominator
    :return: Array of the SMAPE results
    """

    # create array fpr the SMAPE results
    smape_results = np.zeros(shape=(pred_array.shape[1], pred_array.shape[2]))
    # calculate SMAPE for each gridcell
    for row in range(pred_array.shape[1]):
        for col in range(pred_array.shape[2]):
            smape_results[row, col] = calculate_smape(
                real_array[:, row, col],
                pred_array[:, row, col], offset
            )

    return smape_results


def calculate_smape(actual, predicted, offset):
    """
    Calculates the SMAPE for a single timeseries.
    :param actual: true values
    :param predicted: predicted values
    :param offset: offset which is added to the denominator
    :return: SMAPE
    """
    return np.mean(np.abs((actual - predicted) / (actual + predicted + offset))) * 100


def change_predictions_in_cells_below_ceiling_to_zero(train_array, pred_array, ceiling):
    """
    Sets prediction values to zero for cells where the average rides are below the ceiling.
    :param train_array: Array of the training data
    :param pred_array: Predictions
    :param ceiling: Minimum average rides
    :return: Modified array
    """
    train_array_mean = np.mean(train_array, axis=0)
    train_array_mean = np.repeat(train_array_mean[np.newaxis, :, :], pred_array.shape[0], axis=0)

    pred_array = np.where(train_array_mean < ceiling, 0, pred_array)

    return pred_array


def calculate_absolute_error(y_true, y_predicted):
    """
    Calculate absolute error. (predicted - true)
    :param y_true: True values
    :param y_predicted: Predicted values
    :return: absolute error. (predicted - true)
    """
    res_array = y_predicted - y_true
    res_array = res_array.flatten()
    return res_array


def log_grid_error(logger, filename_prediction, grid_error):
    """
    Log error of selected Info model
    :param logger: logger which will output the error
    :param filename_prediction: Filename of the prediction model
    :param grid_error: Tuple containing the info_errors of a model
    """
    # Print Info Predictions
    logger.info(f'######## Prediction errors - {filename_prediction} ########')
    logger.info(f'Average grid RMSE above ceiling: {grid_error[0]}')
    logger.info(f'Average grid MAE above ceiling: {grid_error[1]}')
    logger.info(f'Average grid MAPEC above ceiling: {grid_error[2]}')
    logger.info(f'Global RMSE: {grid_error[3]}')
    logger.info(f'Global MAE: {grid_error[4]}')
    logger.info(f'Global MAPE: {grid_error[5]}')


def log_citywide_error(logger, filename_prediction, grid_error):
    """
    Log error of selected Info model
    :param logger: logger which will output the error
    :param filename_prediction: Filename of the prediction model
    :param grid_error: Tuple containing the info_errors of a model
    """
    # Print Info Predictions
    logger.info(f'######## Prediction errors - {filename_prediction} ########')
    logger.info(f'Global RMSE: {grid_error[0]}')
    logger.info(f'Global MAE: {grid_error[1]}')
    logger.info(f'Global MAPE: {grid_error[2]}')
