from trainer import Trainer


def main():
	data_path = './datasets/dataset.csv'
	param_path = './src/parameter.json'
	trainer = Trainer(data_path=data_path, param_path=param_path)
	trainer.train_and_eval()


if __name__ == '__main__':
	main()