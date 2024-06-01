from transformers import AutoTokenizer


def incremental_decode(tokenizer: AutoTokenizer, token_ids: list):
    # 결과를 저장할 리스트
    valid_strings = []
    temp_ids = []

    for token_id in token_ids:
        # 현재 토큰 ID 추가
        temp_ids.append(token_id)
        # 시도해보고, 성공하면 결과 업데이트
        try:
            decoded = tokenizer.decode(temp_ids)
            if "�" not in decoded:  # replacement character가 없는 경우에만 결과에 추가
                valid_strings.append(decoded)
                temp_ids = []  # 다음 조합을 위해 temp_ids 초기화
        except UnicodeDecodeError:
            # 실패할 경우 그냥 계속 진행 (다음 토큰 ID를 시도)
            continue

    return valid_strings
