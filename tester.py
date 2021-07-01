from time import time

import torch

from utils.utils import generate_piwivos_distance_matrix, probability_to_prediction, resize_tensor, \
    compare_two_frames_k_avg
from utils.input_output import save_mask_test, load_model
from utils.evaluation_metrics import eval_metrics


class Tester:
    def __init__(self, device, model, test_loader, args):
        self.device = device
        self.model = model
        self.test_loader = test_loader
        self.args = args

        checkpoint_path = args.checkpoint_path
        model_state_dict = load_model(checkpoint_path)
        self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(self.device)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            print(f'Testing model using checkpoint {self.args.checkpoint_path}...')
            since = time()
            test_J = []
            test_F = []
            all_time = 0
            all_frames = 0
            all_fps = []

            distance_matrix = generate_piwivos_distance_matrix(self.args.model_name).to(self.device)
            for ii, (frames, masks, info) in enumerate(self.test_loader):
                seq_name = info['name'][0]
                n_frames = info['n_frames'][0].item()
                n_objects = info['n_objects'][0].item()
                original_shape = tuple([x.item() for x in info['original_shape']])
                has_gt = info['has_gt'][0]
                palette = [x.item() for x in info['palette'][0]]

                sequence_time = 0
                since_frame_0 = time()

                # Move frame 0 to GPU and forward pass it
                frame_0 = frames[:, 0]
                _, ch, h, w = frame_0.shape
                frame_0 = frame_0.to(self.device)
                frame_0 = self.model(frame_0)  # (1, ch, h, w)

                _, n_features, low_res_h, low_res_w = frame_0.shape

                # Reduce frame_0 masks and move to GPU
                # (batch, ch=1, h, w) -> (batch, ch=1, h', w')
                masks_0 = resize_tensor(masks[:, 0], low_res_h, low_res_w)
                masks_0 = masks_0.to(self.device)

                # frame_prev is frame_0 for first iteration
                frame_prev = frame_0
                masks_prev = masks_0

                sequence_time += time() - since_frame_0

                # Sequence metrics
                seq_dav_j = torch.empty(0)
                seq_dav_f = torch.empty(0)

                for t in range(1, n_frames):
                    since_frame_t_forward = time()
                    # Select frame_t, move to GPU and forward pass
                    frame_t = frames[:, t]
                    frame_t = frame_t.to(self.device)
                    frame_t = self.model(frame_t)

                    # Obtain scores vs frame_0 and frame_prev
                    scores_0, has_data_0 = compare_two_frames_k_avg(frame_0, frame_t, masks_0, n_objects,
                                                                    self.args.k[0], self.args.lambd[0],
                                                                    distance_matrix)

                    scores_prev, has_data_prev = compare_two_frames_k_avg(frame_prev, frame_t, masks_prev,
                                                                          n_objects, self.args.k[1],
                                                                          self.args.lambd[1], distance_matrix)
                    sequence_time += time() - since_frame_t_forward

                    # Generate low_res_mask, which will be used as masks_prev for following iteration
                    since_frame_t_low = time()
                    probabilities_low = torch.cat((scores_0, scores_prev), dim=1)
                    predicted_masks_low = probability_to_prediction(probabilities_low, n_objects)
                    sequence_time += time() - since_frame_t_low

                    # ## Prediction and metrics ##
                    since_frame_t_pred = time()
                    # Upscale score volumes to original dimensions
                    scores_0 = resize_tensor(scores_0, original_shape[1], original_shape[0], mode='bilinear',
                                             align_corners=True)
                    scores_prev = resize_tensor(scores_prev, original_shape[1], original_shape[0], mode='bilinear',
                                                align_corners=True)

                    # Merge both scores
                    probabilities = torch.cat((scores_0, scores_prev), dim=1)
                    predicted_masks = probability_to_prediction(probabilities, n_objects)

                    sequence_time += time() - since_frame_t_pred

                    if has_gt:
                        # Move gt masks to GPU to compute metrics
                        masks_t = masks[:, t]
                        gt_masks = resize_tensor(masks_t, original_shape[1], original_shape[0])
                        gt_masks = gt_masks.to(self.device)

                        # Compute metrics
                        frame_J, _, frame_F = eval_metrics(predicted_masks, gt_masks, n_objects)
                        seq_dav_j = torch.cat((seq_dav_j, frame_J), dim=0)
                        seq_dav_f = torch.cat((seq_dav_f, frame_F), dim=0)

                    if self.args.export:
                        save_mask_test(predicted_masks, seq_name, t, palette, self.args.checkpoint_path,
                                       self.args.image_set)

                    frame_prev = frame_t
                    masks_prev = predicted_masks_low

                all_time += sequence_time
                all_frames += n_frames
                seq_fps = n_frames / sequence_time
                all_fps.append(seq_fps)

                if has_gt:
                    per_object_j = seq_dav_j.mean(0)
                    per_object_f = seq_dav_f.mean(0)

                    seq_dav_j = seq_dav_j.mean().item()
                    seq_dav_f = seq_dav_f.mean().item()

                    test_J.append(seq_dav_j)
                    test_F.append(seq_dav_f)

                    print(f'{seq_name:<20} | '
                          f'J: {100 * seq_dav_j:>5.2f} % | '
                          f'F: {100 * seq_dav_f:>5.2f} % | '
                          f'G: {50 * seq_dav_j + 50 * seq_dav_f:>5.2f} % | FPS: {seq_fps:5.2f}')
                    for obj in range(per_object_j.shape[0]):
                        print(f'\t\tObject #{obj:2}   | '
                              f'J: {100 * per_object_j[obj]:>5.2f} % | '
                              f'F: {100 * per_object_f[obj]:>5.2f} % | '
                              f'G: {50 * per_object_j[obj] + 50 * per_object_f[obj]:>5.2f} %')
                    print()
                else:
                    print(f'{seq_name:<20} | FPS: {seq_fps:5.2f}')

            fps_mean = sum(all_fps) / len(all_fps)
            fps_real = all_frames / all_time
            print()

            if has_gt:
                test_J = sum(test_J) / len(test_J)
                test_F = sum(test_F) / len(test_F)
                print(f"Testing complted. Elapsed time: {int(time() - since) // 60}'")
                print(f"J: {100 * test_J:.3f} % | F: {100 * test_F:.3f} % | G: {50 * test_J + 50 * test_F:.3f} %")
                print(f"FPS real: {fps_real:.3f} | FPS mean: {fps_mean:.3f}")
            else:
                print(f"Testing complted. Elapsed time: {int(time() - since) // 60}'")
                print(f"FPS real: {fps_real:.3f} | FPS mean: {fps_mean:.3f}")
