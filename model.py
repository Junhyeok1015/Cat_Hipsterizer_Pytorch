import torch
import torchvision.models as models

#mobilenet_v2 model load
# model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)

img_size = 224

mode = 'bbs' # [bbs, lmks]
if mode == 'bbs':
  output_size = 4
elif mode == 'lmks':
  output_size = 18

# # pytorch model
# mobilenetv2_model = models.mobilenet_v2(pretrained=True)

# inputs = Input(shape=(img_size, img_size, 3))



# mobilenetv2_model = MobileNetV2(input_shape=(img_size, img_size, 3), alpha=1.0, include_top=False, weights='imagenet', input_tensor=inputs, pooling='max')

# net = Dense(128, activation='relu')(mobilenetv2_model.layers[-1].output)
# net = Dense(64, activation='relu')(net)
# net = Dense(output_size, activation='linear')(net)

# model = Model(inputs=inputs, outputs=net)

# model.summary() 

class cat_hipsterizer_model(torch.nn.Module):
    """
    cat_hipsterrizer network - encapsulates the base mobilenet v2 network and prediction network
    """
    def __init__(self, num_output):
        super(cat_hipsterizer_model,self).__init__()

        # load pretrained mobilenetv2
        self.model = models.mobilenet_v2(pretrained=True)

        self.droput = torch.nn.Dropout(p=0.2, inplace=False)
        self.fc1 = torch.nn.Linear(1280, 128, bias = True)
        self.fc2 = torch.nn.Linear(128, 64, bias=True)
        self.fc3 = torch.nn.Linear(64, num_output, bias = True)

        self.activation = torch.nn.ReLU()
        self.metric = 0 # used for learning rate policy 'plateau'

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        out = self.model.features(x) # (batch, 1280, 7, 7)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).reshape(out.shape[0], -1)
        out = self.droput(out)
        out = self.fc1(out) # (1280, 128)
        out = self.activation(out)
        out = self.fc2(out) # (128, 64)
        out = self.activation(out)
        out = self.fc3(out)

        return out