import torchvision

def get_network(name):
    network = {
        "VGG16": torchvision.models.vgg16(weights=None),
        "VGG16_bn": torchvision.models.vgg16_bn(weights=None),
        "resnet18": torchvision.models.resnet18(weights=None),
        "resnet34": torchvision.models.resnet34(weights=None),
        "resnet50": torchvision.models.resnet50(weights=None),
	"resnet101": torchvision.models.resnet101(weights=None),
	"resnet152": torchvision.models.resnet152(weights=None),
    }
    if name not in network.keys():
        raise KeyError(f"{name} is not a valid network architecture")
    return network[name]
