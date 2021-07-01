import torch
import torch.nn.functional as F


def compare_two_frames_k_avg(ref, target, ref_masks, max_n_objects, k=3, lambd=0.5, distance_matrix=None):
    """
    Compares the features extracted for the reference frame and the target frame
    :param ref: Features of the reference frame (batch, ch, h, w)
    :param target: Features of the target frame (batch, ch, h, w)
    :param ref_masks: Masks of the reference frame (batch, ch=1, h, w)
    :param max_n_objects: Max number of objects in whole batch
    :param k: Averaging parameter
    :param lambd: Distance parameter
    :param distance_matrix:
    :return scores: Tensor of logit scores for each pixel to belong to each object (batch, max_n_objects, h, w)
    :return data_mask: Tensor with 1s where valid data (batch, max_n_objects)
    """

    # If batch_size != 1 and testing, frame_0 is the same for all frame_t --> expand to batch_size
    if ref.shape != target.shape:
        ref = ref.expand(target.shape)

    b, ch, h, w = ref.shape

    # L2-norm every pixel-wise feature vector
    ref = normalize(ref, 1)
    target = normalize(target, 1)

    # Flatten spatial dimensions to perform pixel-wise comparison
    ref = ref.view(b, ch, h * w)
    target = target.view(b, ch, h * w)

    # Perform pixel-wise comparison
    target = torch.transpose(target, 1, 2)
    pix_scores = torch.bmm(target, ref)
    pix_scores = pix_scores.view(b, h * w, h, w)  # ch: target pixel, (h,w): ref

    # Distance matrix contains a penalization as a function of the distance between the two pixels compared
    if distance_matrix is not None:
        distance_matrix = distance_matrix.unsqueeze(0).expand(pix_scores.shape)
        pix_scores = pix_scores - lambd*distance_matrix

    # One-Hot encoding for masks --> (batch, object_n, h, w)
    oh_masks, has_data = masks_to_oh(ref_masks, max_n_objects)

    # We work with 5 dim to operate: (batch, target_pix, object, h, w)
    pix_scores = pix_scores.unsqueeze(2)
    oh_masks = oh_masks.unsqueeze(1)

    # We define the areas where each object is in the reference frame
    pix_scores_masked = torch.mul(pix_scores, oh_masks.float())

    # Obtain the score of every target pixel belonging to each object
    # Computing the mean of the k most similar pixel-wise scores with reference pixels of that object
    scores = mean_avoid_0(torch.topk(pix_scores_masked.view(b, h * w, max_n_objects, h * w), k=k, dim=-1, largest=True, sorted=False)[0])
    # assert scores.dtype == pix_scores_masked.dtype

    return scores.permute(0, 2, 1).view(b, max_n_objects, h, w), has_data


def mean_avoid_0(inp, dim=-1, epsilon=1e-10):
    """
    Calculates the mean of inp tensor in dimension dim, obviating 0 (not counting them in the number of values)
    :param inp: input tensor
    :param dim: dimension where mean has to be applied
    :param epsilon: to avoid NaNs
    :return: mean
    """
    x = torch.sum(inp, dim=dim)
    y = torch.sum(inp != 0, dim=dim).float()
    return x/(y+epsilon)


def normalize(tensor, dim, epsilon=1e-10):
    """
    Normalizes a tensor so it has L2 norm = 1
    :param tensor: tensor to normalize
    :param dim: dimension of the tensor where normalization will be applied
    :param epsilon: avoid NaNs
    :return: tensor L2 normalized in dimension dim
    """
    x = torch.pow(tensor, 2)
    x = torch.sum(x, dim, True)
    x = x.expand(tensor.shape)
    x = torch.pow(x, 0.5)
    return tensor/(x + epsilon)


def masks_to_oh(masks, max_n_objects):
    """
    Converts a mask in Palette mode to One Hot encoding and generates a mask for the data_tensor where entries are valid
    :param masks: tensor of dim (batch, ch=1, h, w)
    :param max_n_objects:
    :return: tensor of dim (batch, max_num_objects, h, w) with OneHot encoded masks
    :return: boolean tensor of dim (batch, max_num_objects) indicating whether each object is present in the frame
    """
    b, ch, h, w = masks.shape

    # assert max_n_objects == torch.max(masks).item() + 1, 'Això no ho hauria de fallar mai en 0 però pot en PREV'

    objects = torch.arange(max_n_objects).to(masks.device).type(masks.dtype)
    objects = objects.view(1, -1, 1, 1).expand(b, -1, h, w)
    masks = masks.expand(b, max_n_objects, h, w)
    oh_masks = masks.eq(objects).type(masks.dtype)
    has_data = oh_masks.view(b, max_n_objects, -1).byte().any(dim=-1)
    return oh_masks, has_data


def resize_tensor(tensor, h, w, mode='nearest', align_corners=False):
    """
    Resizes the tensor according to mode
    :param tensor: tensor to be resized
    :param h:
    :param w:
    :param mode:
    :param align_corners:
    :return:
    """
    if mode != 'nearest':
        return F.interpolate(tensor.float(), size=(h, w), mode=mode, align_corners=align_corners).type(tensor.dtype)
    else:
        return F.interpolate(tensor.float(), size=(h, w), mode=mode).type(tensor.dtype)


def probability_to_prediction(probs, max_n_objects):
    """
    Computes the final pixel-wise prediction
    :param probs: (batch, max_n_objects*2, h, w)
    :param max_n_objects:
    :return:
    """
    return torch.argmax(probs, 1, True) % max_n_objects  # B1HW


def masked_softmax(logits, has_data=None, dim=1, epsilon=1e-10):
    """
    Normalizes only the masked (with a 1) logits
    :param logits: logits tensor (batch, max_n_objects, h, w)
    :param has_data: mask tensor (batch, max_n_objects)
    :param dim: dimension where the softmax has to be applied
    :param epsilon: to avoid infinities
    :return: tensor of logits' shape with normalized scores
    """
    if has_data is None:
        has_data = torch.ones(logits.shape).to(logits.device)
    elif has_data.shape != logits.shape:
        has_data = has_data.unsqueeze(-1).unsqueeze(-1).expand(logits.shape)

    exps = torch.exp(logits)
    masked_exps = exps * has_data.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps/masked_sums


def masked_weighted_cross_entropy_loss(probabilities, masks, has_data, weighted=True, epsilon=1e-10):
    """
    Takes the normalized scores and calculates the CEL
    :param probabilities: probabilities as a tensor (batch, max_n_objects, h, w)
    :param masks: gt masks (batch, 1, h, w)
    :param has_data: (batch, max_n_objects)
    :param weighted:
    :param epsilon: to avoid infinities
    :return: CEL
    """
    b, max_n_objects, h, w = probabilities.shape

    masks_reshaped = masks.view(b, h*w, 1).long()
    probabilities = torch.log(probabilities.view(b, max_n_objects, h * w).permute(0, 2, 1) + epsilon)

    losses = probabilities.gather(-1, masks_reshaped)
    if weighted:
        weights = calculate_loss_weights(masks, has_data)
        losses = losses * weights.unsqueeze(1).expand(b, h*w, max_n_objects).gather(-1, masks_reshaped)
    return -torch.mean(losses)


def calculate_loss_weights(masks, has_data):
    """
    Calculates weights for Weighted CEL
    :param masks: gt masks (batch, ch=1, h, w)
    :param has_data: tensor that indicates where valid data is (batch, max_n_objects)
    :return weights: weights as 1/f_i, where f_i is the frequency of the ith object in a frame
    """
    max_n_obj = has_data.shape[1]
    b, _, h, w = masks.shape

    objects = torch.arange(max_n_obj).to(masks.device).type(masks.dtype)

    masks = masks.view(b, 1, -1).expand(b, max_n_obj, -1)
    objects = objects.unsqueeze(0).unsqueeze(2).expand(b, -1, h*w)
    weights = torch.sum(masks.eq(objects), dim=-1)

    weights = 1/(weights.float()+1e-10)
    weights = weights * has_data.float()
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    # assert weights.shape == (b, max_n_obj)
    return weights


def generate_piwivos_distance_matrix(model_name='piwivos'):
    if model_name == 'piwivos':
        return _generate_distance_matrix(60, 108)
    elif model_name == 'piwivosf':
        return _generate_distance_matrix(30, 54)
    else:
        raise ValueError(f'model_name should be one of (piwivos, piwivosf), got {model_name}')


def _generate_distance_matrix(h, w):
    """
    Computes the matrix used for pixel-to-pixel distance penalization
    :param h:
    :param w:
    :return m: matrix of dimensions (h*w, h, w)
    """
    hs = torch.arange(h).unsqueeze(-1).expand(h, w)
    ws = torch.arange(w).unsqueeze(0).expand(h, w)
    ref = torch.stack((hs, ws), dim=0).unsqueeze(0).expand(h * w, 2, h, w)

    hs = torch.arange(h).unsqueeze(-1).expand(h, w).reshape(-1).unsqueeze(-1).unsqueeze(-1).expand(h * w, h, w)
    ws = torch.arange(w).repeat(h).unsqueeze(-1).unsqueeze(-1).expand(h * w, h, w)

    target = torch.stack((hs, ws), dim=1)

    m = (ref - target).pow(2).sum(dim=1).pow(0.5)
    m /= m.max()

    return m
