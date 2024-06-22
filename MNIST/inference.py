import torch
from train import FeedForwardNet, download_mnist_datasets

class_mapping = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def predict(model, input, target, class_mapping):
    model.eval()    
    # using a context manager so model doesn't calculate any gradients
    with torch.no_grad():
        predictions = model(input)
        # predictions are of object Tensor (1, 10)
        # confidence levels where sums are 1 (softmax)
        # want to get the max value (most likely confidence)
        predicted_index = predictions[0].argmax()
        # map this to the relative class mapping
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    # 1. load back the model
    feed_forward_net = FeedForwardNet()
    state_dict = torch.load("feedforwardnet.pth")
    feed_forward_net.load_state_dict(state_dict)

    # 2. load MNIST test dataset
    _, test_data = download_mnist_datasets()

    # 3. get sample from test dataset for inference (input and target)
    input, target = test_data[0][0], test_data[0][1]

    # 4. make an inference 
    predicted, expected = predict(feed_forward_net, input, target, class_mapping)

    print(f"Predicted: {predicted}, Expected: '{expected}'")
