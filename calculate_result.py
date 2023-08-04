import numpy as np

from yaml_utils import read_yaml_to_dict


# extract result from yaml file

def calculate_result(seeds, result_name, result):
    # Extract information for each seed and then average it
    train_R2_list = []
    train_RMSEC_list = []
    train_MSE_list = []
    train_MAE_list = []
    test_R2_list = []
    test_RMSEP_list = []
    test_MSE_list = []
    test_MAE_list = []
    SDR_list = []
    for seed in seeds:
        item_result = result['seed_' + str(seed)][result_name]
        train_R2_list.append(float(item_result['train R2']))
        train_RMSEC_list.append(float(item_result['train RMSEC']))
        train_MSE_list.append(float(item_result['train MSE']))
        train_MAE_list.append(float(item_result['train MAE']))

        test_R2_list.append(float(item_result['test R2']))
        test_RMSEP_list.append(float(item_result['test RMSEP']))
        test_MSE_list.append(float(item_result['test MSE']))
        test_MAE_list.append(float(item_result['test MAE']))
        SDR_list.append(float(item_result['SDR']))

    # Calculate the mean and standard deviation of various indicators
    print('train R2: {:.4f}±{:.4f}'.format(np.mean(train_R2_list), np.std(train_R2_list)))
    print('train RMSEC: {:.4f}±{:.4f}'.format(np.mean(train_RMSEC_list), np.std(train_RMSEC_list)))
    print('train MSE: {:.4f}±{:.4f}'.format(np.mean(train_MSE_list), np.std(train_MSE_list)))
    print('train MAE: {:.4f}±{:.4f}'.format(np.mean(train_MAE_list), np.std(train_MAE_list)))

    print('test R2: {:.4f}±{:.4f}'.format(np.mean(test_R2_list), np.std(test_R2_list)))
    print('test RMSEP: {:.4f}±{:.4f}'.format(np.mean(test_RMSEP_list), np.std(test_RMSEP_list)))
    print('test MSE: {:.4f}±{:.4f}'.format(np.mean(test_MSE_list), np.std(test_MSE_list)))
    print('test MAE: {:.4f}±{:.4f}'.format(np.mean(test_MAE_list), np.std(test_MAE_list)))
    print('SDR: {:.4f}±{:.4f}'.format(np.mean(SDR_list), np.std(SDR_list)))


if __name__ == '__main__':
    seeds = [1300, 1301, 1303, 1304, 1305, 1306, 1308, 1309, 1313, 1317]

    yaml_path = 'logger/result_MBA/alpha=0_dot_2.yaml'

    result = read_yaml_to_dict(yaml_path)

    result_names = ["model_result"]

    for result_name in result_names:
        print(result_name)
        calculate_result(seeds, result_name, result)
        print()
