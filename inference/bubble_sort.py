from typing import Union, List


def sorter(arr: Union[List[int], List[float]]) -> Union[List[int], List[float]]:
    # In-place insertion sort (faster than bubble sort for nearly sorted or small arrays)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
