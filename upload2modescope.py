from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'ms-25733ddb-b154-4568-bdfb-097f268b4f44'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'gengxuelong/'
dataset_name = 'OSUM-gender-data'

api.upload_folder(
    repo_id=f"{owner_name}/{dataset_name}",
    folder_path='/path/to/local/dir',
    commit_message='upload dataset folder to repo',
    repo_type = 'dataset'
)