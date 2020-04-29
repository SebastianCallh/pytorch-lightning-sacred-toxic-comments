from sacred.observers import MongoObserver, RunObserver


def expexp_observer(user: str, password: str) -> RunObserver:
    host = 'expexp.westeurope.cloudapp.azure.com'
    port = 27017
    database = 'testdb'
    url = f'mongodb://{user}:{password}@{host}:{port}/?authSource=admin'
    return MongoObserver(url, db_name=database)
