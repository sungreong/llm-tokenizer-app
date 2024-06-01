from transformers import AutoTokenizer
from utils.decoder import incremental_decode

tokenizer_name = "DWDMaiMai/tiktoken_cl100k_base"
# tokenizer_name = "bert-base-uncased"
# tokenizer_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
)
text = "안녕하세요"
text = text.encode("utf-8").decode("utf-8")

encoded = tokenizer.encode(text)

# 텍스트 인코딩
text = "안녕하세요"
tokenizer.tokenize(text)
encoded = tokenizer.encode(text)
print("인코딩된 토큰들:", encoded)
tokenizer.convert_ids_to_tokens(encoded)
# 함수 호출하여 디코딩된 문자열 리스트 얻기
decoded_strings = incremental_decode(tokenizer, encoded)
print("디코딩된 문자열들:", decoded_strings)

print(len(decoded_strings), len(encoded))


from typing import Optional

from huggingface_hub import get_hf_file_metadata, hf_hub_url, repo_info
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError


def repo_exists(repo_id: str, repo_type: Optional[str] = None, token: Optional[str] = None) -> bool:
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except RepositoryNotFoundError:
        return False
