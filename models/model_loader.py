from models import ablation_1, ablation_2, MBA


def get_model(args, data_info):
    if args.model == 'ablation_1':
        model = ablation_1.create_model(args, data_info)
    elif args.model == 'ablation_2':
        model = ablation_2.create_model(args, data_info)
    elif args.model == 'MBA':
        model = MBA.create_model(args, data_info)
    elif args.model == 'xx':
        # add other model
        # model = xx.create_model(args, data_info)
        pass
    else:
        print('Select model is not exist,Please checking!')

    return model
