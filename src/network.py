from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
import config.config as opt


class CNNNetwork(Module):

    def __init__(self):
        """
        Function containing the CNN network with the Conv layers and linear layers.

        Parameters: self - instance of the class


        Returns:
                Creates CNN layer and linear layers

                            """

        super(CNNNetwork, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 4, kernel_size=3, stride=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(4, 8, kernel_size=3, stride=1),
            BatchNorm2d(8),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(8, 16, kernel_size=3, stride=1),
            BatchNorm2d(16),
            ReLU(inplace=True),

        )

        self.linear_layers = Sequential(

            Linear(16*3*3, opt.output_size)

        )

    def forward_pass(self, x):
        """
        Function to execute forward propagation in the CNN network.

        Parameters:
                self - instance of the class
                x - Input image data

        Returns:
                Final model output label

        """

        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x
