from prefect import task
from functools import wraps, partial
from prefect.backend.artifacts import create_markdown_artifact
import pandas as pd  

def artifact_task(func=None, **task_init_kwargs):
  
    if func is None: 
        return partial(artifact_task, **task_init_kwargs)

    @wraps(func)
    def safe_func(**kwargs):
        res = func(**kwargs)
        if isinstance(res, pd.DataFrame):
            create_markdown_artifact(res.head(10).to_markdown())
        return res

    safe_func.__name__ = func.__name__
    
    return task(safe_func, **task_init_kwargs)