import torchvision.transforms as standard_transforms
from .GTA_Events import GTA_Events

def loading_gta_events_data(data_root):
    # the pre-proccssing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
    ])
    # create the training dataset
    train_set = GTA_Events(data_root, train=True, transform=transform, patch=False, flip=False)
    # create the validation dataset
    val_set = GTA_Events(data_root, train=False, transform=transform)

    return train_set, val_set

# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'GTA_EVENTS':
        from dataset import loading_gta_events_data
        return loading_gta_events_data

    return None