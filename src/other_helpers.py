from .models import MinimalSource


def get_chunk_text(source: MinimalSource) -> str:
    with open(source.file_path) as f:
        content = f.read()
    return content[
        source.first_character_index:source.last_character_index
    ]


def is_match(source1: MinimalSource, source2: MinimalSource) -> bool:
    if source1.file_path != source2.file_path:
        return False
    intersection = max(
        0,
        min(source1.last_character_index, source2.last_character_index)
        - max(source1.first_character_index, source2.first_character_index)
    )
    union = (
        max(source1.last_character_index, source2.last_character_index)
        - min(source1.first_character_index, source2.first_character_index)
    )
    return union > 0 and intersection / union >= 0.05
