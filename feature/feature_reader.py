"""

Warning:
    The `FeatureReader` is not thread-safety
    because we will use a dict to make the number of file-handler limited
    If you use the same `FeatureReader` instance in different process or thread,
    it could cause mistakes!
    (For example, if you use FeatureReader in torch.data.util.Dataset and set the Dataloader.num_workers > 1,
    it will cause runtime error!)
"""


class FeatureReader():

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


def main():
    feats_scp_file = ""

    return


if __name__ == '__main__':
    main()
