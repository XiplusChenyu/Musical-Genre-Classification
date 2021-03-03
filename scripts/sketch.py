def partition(A, l, r):
    pivot = A[r]
    i = l - 1
    for j in range(l, r):
        if A[j] <= pivot:
            i += 1
            A[i], A[j] = A[j], A[i]
    A[i+1], A[r] = A[r], A[i+1]
    return i+1

def quickSort(A, l, r):
    if l < r:
        dividor = partition(A, l, r)
