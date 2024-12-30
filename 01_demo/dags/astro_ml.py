from airflow.decorators import dag
from datetime import datetime
from astro import sql as aql
from astro.files import File
from astro.dataframes.pandas import DataFrame
import os


@dag(
    dag_id="astro_ml", 
    schedule_interval=None, 
    start_date=datetime(2024, 1, 1), 
    catchup=False)
def astro_ml():
    data_bucket = "file/data"
    model_bucket = "file/model"

    model_id = datetime.utcnow().strftime("%y_%d_%m_%H_%M_%S_%f")
    model_dir = os.path.join(model_bucket, model_id)

    @aql.dataframe(task_id='extract')
    def extract_housing_data() -> DataFrame:
        from sklearn.datasets import fetch_california_housing
        import pandas as pd
        import os

        # Get data
        housing_data = fetch_california_housing(download_if_missing=True, as_frame=True).frame

        # Save to temporary CSV file
        os.makedirs('file/data/raw', exist_ok=True)
        temp_path = 'file/data/raw/housing_data.csv'
        housing_data.to_csv(temp_path, index=False)        

        # Read back as DataFrame
        return pd.read_csv(temp_path)

    @aql.dataframe(task_id='featurize')
    def build_features(raw_df:DataFrame, model_dir:str) -> DataFrame:
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        from joblib import dump
        import os
       
        target = 'MedHouseVal'
        x = raw_df.drop(target, axis=1)
        y = raw_df[target]
        scaler = StandardScaler()
        x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)
        metrics_df = pd.DataFrame(scaler.mean_, index=x.columns)[0].to_dict()

        #Save scalar for later monitoring and eval
        os.makedirs(model_dir, exist_ok=True)
        dump([metrics_df, scaler], os.path.join(model_dir, 'scaler.joblib'))

        x[target] = y
        return x
    
    @aql.dataframe(task_id='train')
    def train_model(feature_df:DataFrame, model_dir:str) -> str:
        from sklearn.linear_model import RidgeCV
        import numpy as np
        from joblib import dump
        import os

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        target = 'MedHouseVal'
        model = RidgeCV(alphas=np.logspace(-3, 1, num=30))
        reg = model.fit(feature_df.drop(target, axis=1), feature_df[target])

        model_file_path = model_dir +  '/ridgecv.joblib'
        dump(reg, model_file_path)
        print(f"Model saved to {model_file_path}") 

        
        return model_file_path
    
    #Score data
    @aql.dataframe(task_id='predict')
    def predict_housing(feature_df:DataFrame, model_file_path:str) -> DataFrame:
        from joblib import load

        loaded_model = load(model_file_path)

        target = 'MedHouseVal'

        feature_df['preds'] = loaded_model.predict(feature_df.drop(target, axis=1))
        print(feature_df)

        return feature_df
    
    extract_df = extract_housing_data()
    loaded_data = aql.export_file(
        task_id='save_data',
        input_data=extract_df,
        output_file=File(os.path.join(data_bucket, 'housing.csv')),
        if_exists='replace')
    feature_df = build_features(extract_df, model_dir)
    model_file_uri = train_model(feature_df, model_dir)
    pred_df = predict_housing(feature_df, model_file_uri)

    pred_file = aql.export_file(
        task_id='save_predictions',
        input_data=pred_df,
        output_file=File(os.path.join(data_bucket, 'predictions.csv')),
        if_exists='replace')
    
astro_ml()