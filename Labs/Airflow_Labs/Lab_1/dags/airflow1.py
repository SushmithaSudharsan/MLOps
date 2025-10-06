# Import in DAG file
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow, save_cluster_assignments
import pickle
import base64
import os

# Define DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Airflow Lab 1 DAG with cluster assignments',
    catchup=False,
) as dag:

    # Existing tasks
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

    save_clusters_task = PythonOperator(
        task_id="save_clusters_task",
        python_callable=save_cluster_assignments,
    )
    # -------------------
    # Set dependencies
    # -------------------
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task >> save_clusters_task
