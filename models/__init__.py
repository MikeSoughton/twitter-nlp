"""
When this module function is called, it will get the models in the files corresponding to
the appropriate model name. 

Models can be added to this directory in the future.
"""

def get_model(model_name, conv_layers = 2, max_dilation_rate = 4, max_features = 20000, max_len = 100):
    if model_name == 'solo_cnn':
        from .solo_cnn import model
        model = model(conv_layers = conv_layers, max_dilation_rate = max_dilation_rate, max_features = max_features, max_len = max_len)

    return model