from arg_parser import training_parser
from training import training_loop

def main():
	args = training_parser().parse_args()

	LR_G             = args.learning_rate
	EPOCHS           = args.epochs
	BATCH_SIZE       = args.batch_size
	N_TRAINING_DATA  = args.n_training_data
	training_loop(LR_G, EPOCHS, BATCH_SIZE, N_TRAINING_DATA)



if __name__ == '__main__':
    main()
