import torch
import matplotlib.pyplot as plt
from torchvision import transforms


from sklearn.metrics import balanced_accuracy_score
import imagenet_labels
association_1000 = imagenet_labels.association_1000
association_22k = imagenet_labels.association_22k

from torch.nn.functional import softmax


class UnNormalize(object):
    """
    To undo a image normalisation
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor



unormalize = UnNormalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def evaluate_model_multiclass(model, test_loader):
    """
    Evaluation of multi class model

    Args:
        model : A multi class model (trained on imagenet 22k or imagenet 1k)
        loader : A PyTorch loader to perform evaluation of the model
    Returns:

    """
    test_features, test_labels = next(iter(test_loader))
    outputs = model(test_features)
    preds = outputs.argmax(dim=1)

    for img, label, pred, output in zip(test_features, test_labels, preds, outputs):
        img = img.squeeze()

        img = transforms.functional.to_pil_image(unormalize(img))
        plt.imshow(img)
        plt.show()
        print(f"Label: {label}")

        pred = int(pred.numpy())
        print(pred)

        sim, topk = torch.topk(output, 5)

        if outputs.shape[1] == 1000:
            association = association_1000
        else:
            association = association_22k

        topk = [association[e] for e in topk.numpy()]
        sim = softmax(sim, dim=-1).detach().numpy()

        for label, score in zip(topk, sim):
            print(f"Score : {score:.2f} - Label {label}")


def evaluate_model_monoclass(model, test_loader, th=0):
    """
    Evaluation of mono class model

    Args:
        model : A mono class model
        loader : A PyTorch loader to perform an evaluation of the model
        th : decision threshold
    Returns:
        ba_score : balanced accuracy score over classes
    """
    test_features, test_labels = next(iter(test_loader))
    outputs = model(test_features).detach().numpy().ravel()

    preds = outputs >= th
    preds = preds.astype(int)
    ba_score = balanced_accuracy_score(test_labels, preds)

    for img, label, pred, output in zip(test_features, test_labels, preds, outputs):
        img = img.squeeze()

        img = transforms.functional.to_pil_image(unormalize(img))
        plt.imshow(img)
        plt.show()
        print(f"Label: {label}")
        print(f"Score : {output}")
        print(f"Pred : {pred}")
    return ba_score