from ImageTransfer.slow_art import transfer_train as train

# dir
STYLE_PATH = './data/style/The_Starry_Night.jpg'
CONTENT_PATH = './data/input/chicago.jpg'
SAVE_PATH = './data/output/'

if __name__ == '__main__':
    output_save_path = train.train(CONTENT_PATH, STYLE_PATH, SAVE_PATH)
    print(output_save_path)
