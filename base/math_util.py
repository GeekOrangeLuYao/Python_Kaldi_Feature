# kaldi-math.cc

def RoundUpToNearestPowerOfTwo(n):
    assert n > 0, f"Error in RoundUpToNearestPowerOfTwo, n should be more than 0"
    n -= 1
    n |= (n >> 1)
    n |= (n >> 2)
    n |= (n >> 4)
    n |= (n >> 8)
    n |= (n >> 16)
    return n + 1
