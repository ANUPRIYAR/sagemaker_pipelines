"""Evaluation script for measuring root mean squared error."""
import json
import logging
import pathlib
import tarfile


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    logger.debug("Starting evaluation.")
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    logger.info("Loading jsons.")
    with open("evaluation_metrics.json", "r") as f:
        eval_metrics = json.load(f)
        
    logger.info(eval_metrics)
    
    rmse = eval_metrics['RMSE']
    logger.info(f"rmse = {rmse}")
    
    
    report_dict = {
        "regression_metrics": {
            "rmse": {
                "value": rmse
            },
        },
    }
    # writing evaluation report   
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # logger.info("Writing out evaluation report with mse: %f", mse)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
    
