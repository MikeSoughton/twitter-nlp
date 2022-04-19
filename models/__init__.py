



def build_model(model_name, conv_layers = 2, max_dilation_rate = 4, max_features = 20000, max_len = 100):
    if model_name == 'solo_cnn':
        from .solo_cnn import model
        model = model(conv_layers = conv_layers, max_dilation_rate = max_dilation_rate, max_features = max_features, max_len = max_len)

    return model