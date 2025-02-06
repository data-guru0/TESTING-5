import kfp
from kfp import dsl

# Define your components (steps) here
def data_processing_op():
    return dsl.ContainerOp(
        name="Data Processing",
        image="dataguru97/my-mlops-app:latest",  # Use the Docker image you pushed
        command=["python", "src/data_processing.py"]
    )

def model_training_op():
    return dsl.ContainerOp(
        name="Model Training",
        image="dataguru97/my-mlops-app:latest",  # Same Docker image
        command=["python", "src/model_training.py"]
    )

# Define the pipeline
@dsl.pipeline(
    name="MLops Pipeline",
    description="Pipeline for Data Processing and Model Training"
)
def mlops_pipeline():
    # Create pipeline steps
    data_processing = data_processing_op()
    model_training = model_training_op().after(data_processing)

# Compile the pipeline
if __name__ == "__main__":
    kfp.compiler.Compiler().compile(mlops_pipeline, "mlops_pipeline.yaml")
