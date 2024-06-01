from typing import Optional

from huggingface_hub import get_hf_file_metadata, hf_hub_url, repo_info
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError


def repo_exists(repo_id: str, repo_type: Optional[str] = None, token: Optional[str] = None) -> bool:
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except RepositoryNotFoundError:
        return False
