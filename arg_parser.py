import argparse

def training_parser():
    parser = argparse.ArgumentParser(description='Training arguments.')
    
    parser.add_argument('-lr', '--learning_rate', action='store',
                         default=1e-5, type=float, 
                         help=('Learning Rate. Default: '
                               '0.00001'))
    parser.add_argument('-bs', '--batch_size', action='store', 
                         default=2, type=int, 
                         help='Batch Size. Default: "2"')
    parser.add_argument('-ep', '--epochs', action='store', default=1, 
                         type=int, help=('Epochs. Default: 1'))
    parser.add_argument('-lt', '--loss_type', action='store', default='l2_loss', 
                         type=str, choices=['l1_loss', 'l2_loss'],
                         help=('Loss type, either L1 (MAE) or L2 (MSE). Default: L2'))
    parser.add_argument('-ntd', '--n_training_data', action='store', default=800, 
                         type=int, help=('Number of training data used each epoch. Default: 800'))    

    return parser
