class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return 'data/cityscapes'     # foler that contains leftImg8bit/
        elif dataset == 'loveda':
            return 'data/loveda'
        elif dataset == 'floodnet':
            return 'data/floodnet'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
