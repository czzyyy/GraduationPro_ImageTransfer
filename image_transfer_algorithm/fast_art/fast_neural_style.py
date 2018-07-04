from ImageTransfer.fast_art import transfer_train as train

# dir
STYLE_PATH = './stored_models/test/rain_princess.jpg'
TRAIN_PATH = './data/train2014'
TRY_PATH = './test'

if __name__ == '__main__':
    train.train(content_path=TRY_PATH, style_path=STYLE_PATH)
