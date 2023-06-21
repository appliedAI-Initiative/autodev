from typing import List


def segmentation(seq: List[int], separator: int) -> List[List[int]]:
    result = []
    curr = []
    for i, item in enumerate(seq):
        if item == separator:
            result.append(curr)
            curr = []
        else:
            curr.append(item)
    return result
