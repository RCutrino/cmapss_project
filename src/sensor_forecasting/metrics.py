"""
Evaluation metrics for sensor forecasting.
 
Metrics
-------
- RMSE : Root Mean Squared Error
- MAE  : Mean Absolute Error
- R²   : Coefficient of determination
"""
def evaluate_forecast(model_name, y_true, y_pred, sensor, results):
    rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3)
    mae  = round(float(mean_absolute_error(y_true, y_pred)), 3)
    r2   = round(float(r2_score(y_true, y_pred)), 3)
    results[sensor][model_name] = {'RMSE': rmse, 
                                   'MAE': mae, 
                                   'R²': r2}
    print(f"{sensor:<12} {model_name:<20} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.3f}")

    return results
