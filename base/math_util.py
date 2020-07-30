# kaldi-math.cc

def RoundUpToNearestPowerOfTwo(n):
    # This function is to calc this:
    #  2 -> 2
    #  3 -> 4
    #  6 -> 8
    # 15 -> 16
    # 22 -> 32
    # 65 -> 128
    assert n > 0, f"Error in RoundUpToNearestPowerOfTwo, n should be more than 0"
    n -= 1
    n |= (n >> 1)
    n |= (n >> 2)
    n |= (n >> 4)
    n |= (n >> 8)
    n |= (n >> 16)
    return n + 1



def main():
    n = 65
    print(RoundUpToNearestPowerOfTwo(n))


if __name__ == '__main__':
    main()