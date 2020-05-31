
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 32,
    'ni': 512,
    'enc_nh': 1024,
    'dec_nh': 1024,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'batch_size': 32,
    'epochs': 100,
    'test_nepoch': 5,
    'train_data': 'datasets/yelp_data/yelp.train.txt',
    'val_data': 'datasets/yelp_data/yelp.valid.txt',
    'test_data': 'datasets/yelp_data/yelp.test.txt',
    'label':True
}
