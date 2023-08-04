import argparse


class Constant:
    BRIX = 0

    # test model
    INTERNAL = 'internal'

    # dataset
    blueberry = 'blueberry'

    # internal
    # I'm very sorry, this is only partial data, not all blueberry spectral data
    train_csv_path_list = [
        './data/data_1.csv',
        './data/data_1.csv',
        './data/data_1.csv',
        './data/data_1.csv',
        './data/data_1.csv',
    ]

    # Leave only one interface and do not use this data
    train_mat_path_list = [
        './data/data_image_1.mat',
        './data/data_image_1.mat',
        './data/data_image_1.mat',
        './data/data_image_1.mat',
        './data/data_image_1.mat'
    ]


def parse_args():
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', type=str, default=Constant.INTERNAL,
                        help='Test Mode of the code. "Constant.INTERNAL(internal)" ')

    parser.add_argument('--dataset', type=str, default=Constant.blueberry, help='')

    parser.add_argument('--train_csv_path_list', type=str, nargs='+',
                        default=Constant.train_csv_path_list,
                        help='The train csv data path list.')

    parser.add_argument('--train_mat_path_list', type=str, nargs='+',
                        default=Constant.train_mat_path_list,
                        help='The train mat data path list.')

    parser.add_argument('--train_split_rate', type=float, default=0.8, help='The train ratio used for the train split.')

    parser.add_argument('--refresh_rate', type=int, default=100, help='The refresh_rate of ProgressBar.')

    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the dataloader.')

    parser.add_argument('--model', type=str, default='MBA',
                        help='select use model [ablation_1|ablation_2|MBA]')

    parser.add_argument('--y_is_standard', type=bool, default=True, help='y value whether standardization is required')

    # Model args
    parser.add_argument('--property', type=int, default=Constant.BRIX, help='the property will be predicted.')

    parser.add_argument('--checkpoint', type=str, default='', help='Model checkpoint path.')

    parser.add_argument('--mode', type=str, default='train', help='Mode of the code. Can be set to "train" or "test".')

    parser.add_argument('--epochs', type=int, default=5000, help='The number of epochs.')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate.')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size.')
    parser.add_argument('--test_batch_size', type=int, default=1000, help='The test batch size.')
    parser.add_argument('--weight_decay', type=float, default=0.000, help='The parameter of L2 normalization.')

    # Parse
    args = parser.parse_args()
    # Process args
    return args
