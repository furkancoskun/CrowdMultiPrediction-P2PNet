from .p2pnet_withAnomaly_Headv2 import build_anomaly_headv2
from .p2pnet_withAnomaly import build_anomaly
from .p2pnet import build

# build the P2PNet model
# set training to 'True' during training
def build_model(args, training=False):
    if (args.model_type == "P2PNetWithAnomaly"):
        return build_anomaly(args, training)
    elif (args.model_type == "P2PNetWithAnomaly_Headv2"):
        return build_anomaly_headv2(args, training)        
    elif (args.model_type == "P2PNet"):
        return build(args, training)
    else:
        print("ERROR! Invalid model_type!")
        print("No model_type named " + str(args.model_type))