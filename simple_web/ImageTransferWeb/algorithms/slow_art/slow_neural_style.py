from ImageTransferWeb.algorithms.slow_art import transfer_train as train

# dir
# STYLE_PATH = './data/style/The_Starry_Night.jpg'
# CONTENT_PATH = './data/input/chicago.jpg'
# SAVE_PATH = './data/output/'


def start_slow_neural_style(style_path='./data/style/The_Starry_Night.jpg', content_path='./data/input/chicago.jpg',
                            save_path='./data/output/'):
    output_save_path = train.train(content_path, style_path, save_path)
    return output_save_path
