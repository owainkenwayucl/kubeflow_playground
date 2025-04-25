from kfp import dsl, compiler
from kfp.client import Client

@dsl.component
def say_hello(name: str) -> str:
    hello_text = f'Hello, {name}!'
    print(hello_text)
    return hello_text

@dsl.pipeline
def hello_pipeline(recipient: str) -> str:
    hello_task = say_hello(name=recipient)
    return hello_task.output

compiler.Compiler().compile(hello_pipeline, 'hello_pipeline.yaml')

client = Client(host='<MY-KFP-ENDPOINT>')
run = client.create_run_from_pipeline_package(
    'hello_pipeline.yaml',
    arguments={
        'recipient': 'World',
    },
)