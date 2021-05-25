from kfp import dsl
import kfp

def download_dataset_op():
    return dsl.ContainerOp(
        name="Download Dataset",
        image="mindang241/sentiment-analysis-download-dataset:0.3",
        arguments=[],
        file_outputs={
            'train': "/app/datasets/train.csv",
            'dev': "/app/datasets/dev.csv",
            'test': "/app/datasets/test.csv",
        }
    )

def train_op(train, dev):
    return dsl.ContainerOp(
        name="Train model",
        image="mindang241/sentiment-analysis-train-model:0.1",
        arguments=[
            '--train', train,
            '--dev', dev
        ],
        file_outputs={
            'model': '/app/model.pkl'
        }
    )

def evaluate_op(test, model):
    return dsl.ContainerOp(
        name="Evaluate model",
        image="mindang241/sentiment-analysis-evaluate-model:0.1",
        arguments=[
            '--test', test,
            '--model', model
        ]
    )


@dsl.pipeline(
    name="Sentiment Analysis pipeline",
    description="Sentiment analysis pipeline for UIT-VSFC dataset"
)

def boston_pipeline():
    _download_dataset_op = download_dataset_op()

    # _train_op = train_op(
    #     dsl.InputArgumentPath(_download_dataset_op.outputs['train']),
    #     dsl.InputArgumentPath(_download_dataset_op.outputs['dev'])
    # ).after(_download_dataset_op)

    # _evaluate_op = _evaluate_op(
    #     dsl.InputArgumentPath(_download_dataset_op.outputs['test']),
    #     dsl.InputArgumentPath(_train_op.outputs['model']),
    # ).after(_train_op)

client = kfp.Client(host="http://localhost:3000")
client.create_run_from_pipeline_func(boston_pipeline, arguments={})