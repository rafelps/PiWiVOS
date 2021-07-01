import os
import math
from time import time

import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from utils.utils import generate_piwivos_distance_matrix, probability_to_prediction, masked_softmax, resize_tensor, \
    compare_two_frames_k_avg, masked_weighted_cross_entropy_loss
from utils.input_output import save_model
from utils.evaluation_metrics import my_eval_iou


class Trainer:
    def __init__(self, device, model, train_loader, val_loader, args):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9,
                             weight_decay=args.weight_decay)

        self.best_val_iou = 0

        print(f'Starting training job {args.job_name}...')

        self.summary_writer = SummaryWriter(os.path.join('logs', args.job_name))

    def train_model(self):
        # ########## TRAIN VAL LOOP ##########
        since = time()
        n_train_iterations = int(math.ceil(len(self.train_loader.dataset) / self.train_loader.batch_size))
        distance_matrix = generate_piwivos_distance_matrix(self.args.model_name).to(self.device)

        for epoch in range(self.args.num_epochs):
            since_epoch = time()

            print('Starting Epoch {}/{}'.format(epoch + 1, self.args.num_epochs))
            print()

            # ########## TRAIN LOOP ##########
            print('Training...')
            self.model.train()

            epoch_train_loss = []
            epoch_train_iou = torch.empty(0)

            for ii, (frames, masks, info) in enumerate(self.train_loader):
                n_objects = info['n_objects']  # Including background
                max_n_objects = n_objects.max().item()

                # Move all frames to GPU and forward pass them
                batch_size, n_frames, ch, h, w = frames.shape
                frames = frames.view(-1, ch, h, w)
                frames = frames.to(self.device)
                frames = self.model(frames)
                n_tot_frames, n_features, low_res_h, low_res_w = frames.shape
                frames = frames.view(batch_size, n_frames, n_features, low_res_h, low_res_w)

                # Reduce frame_0 and frame_prev masks and move to GPU
                # (batch, ch=1, h, w) -> (batch, ch=1, h', w')
                masks_0 = resize_tensor(masks[:, 0], low_res_h, low_res_w)
                masks_prev = resize_tensor(masks[:, 1], low_res_h, low_res_w)
                masks_0 = masks_0.to(self.device)
                masks_prev = masks_prev.to(self.device)

                # Obtain scores vs frame_0 and frame_prev
                scores_0, has_data_0 = compare_two_frames_k_avg(frames[:, 0], frames[:, 2], masks_0, max_n_objects,
                                                                self.args.k[0], self.args.lambd[0], distance_matrix)
                # assert torch.sum(has_data_0, dim=-1).cpu() == n_objects,
                # 'Mask reduction has caused a loss of some objects'

                scores_prev, has_data_prev = compare_two_frames_k_avg(frames[:, 1], frames[:, 2], masks_prev,
                                                                      max_n_objects, self.args.k[1], self.args.lambd[1],
                                                                      distance_matrix)

                probabilities_0 = masked_softmax(scores_0, has_data_0)
                probabilities_prev = masked_softmax(scores_prev, has_data_prev)
                probabilities = self.args.weight0 * probabilities_0 + (1 - self.args.weight0) * probabilities_prev

                # Computing loss at low dimensions, backpropagation and optimizer step
                masks_t = resize_tensor(masks[:, 2], low_res_h, low_res_w)
                masks_t = masks_t.to(self.device)

                self.optimizer.zero_grad()
                loss = masked_weighted_cross_entropy_loss(probabilities, masks_t, has_data_0, self.args.weighted_loss)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss.append(loss.item())

                # ## Prediction and metrics ##
                # Upscale score volumes to original dimensions
                scores_0 = resize_tensor(scores_0, h, w, mode='bilinear', align_corners=True)
                scores_prev = resize_tensor(scores_prev, h, w, mode='bilinear', align_corners=True)

                # Merge both scores
                probabilities = torch.cat((scores_0, scores_prev), dim=1)

                predicted_masks = probability_to_prediction(probabilities, max_n_objects)
                gt_masks = masks[:, 2].to(self.device)

                # Compute per-object IoU
                object_ious = my_eval_iou(predicted_masks, gt_masks, max_n_objects)
                # Compute per-frame IoU taking the mean (we use has_data for possible batches with different n_obj)
                ious = (object_ious * has_data_0[:, 1:].float().cpu()).sum(dim=1) / \
                       (has_data_0[:, 1:].float().cpu()).sum(dim=1)
                epoch_train_iou = torch.cat((epoch_train_iou, ious), 0)

                if ((ii + 1) % self.args.log_each == 0) or (ii == n_train_iterations - 1):
                    print(f"Iteration {ii + 1:4}/{n_train_iterations:4} | loss: {loss:.4f}")

            epoch_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
            epoch_train_iou = epoch_train_iou.mean().item()
            self.summary_writer.add_scalar('train_loss', epoch_train_loss, epoch + 1)
            self.summary_writer.add_scalar('train_iou', epoch_train_iou, epoch + 1)
            print()

            # ########## VAL LOOP ##########
            print("Validating...")
            self.model.eval()
            with torch.no_grad():
                epoch_val_loss = []
                epoch_val_iou = []
                # Iterate over sequences
                for ii, (frames, masks, info) in enumerate(self.val_loader):
                    seq_name = info['name'][0]
                    n_frames = info['n_frames'][0].item()
                    n_objects = info['n_objects'][0].item()  # Including background
                    original_shape = tuple([x.item() for x in info['original_shape']])

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
                    sequence_loss = []
                    sequence_iou = []

                    # Iterate over frames
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

                        probabilities_0 = masked_softmax(scores_0, has_data_0)
                        probabilities_prev = masked_softmax(scores_prev, has_data_prev)
                        probabilities = self.args.weight0 * probabilities_0 + \
                                        (1 - self.args.weight0) * probabilities_prev

                        # Computing loss at low dimension
                        masks_t = masks[:, t]
                        masks_t_low = resize_tensor(masks_t, low_res_h, low_res_w)
                        masks_t_low = masks_t_low.to(self.device)

                        loss = masked_weighted_cross_entropy_loss(probabilities, masks_t_low, has_data_0,
                                                                  self.args.weighted_loss)
                        sequence_loss.append(loss.item())

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

                        # Move gt masks to GPU to compute metrics
                        gt_masks = resize_tensor(masks_t, original_shape[1], original_shape[0])
                        gt_masks = gt_masks.to(self.device)

                        # Compute IoU
                        object_ious = my_eval_iou(predicted_masks, gt_masks, n_objects)
                        ious = object_ious.mean().item()
                        sequence_iou.append(ious)

                        # Current frame and mask become prev for next frame
                        frame_prev = frame_t
                        masks_prev = predicted_masks_low

                    # Log metrics
                    sequence_iou = sum(sequence_iou) / len(sequence_iou)
                    epoch_val_iou.append(sequence_iou)
                    sequence_loss = sum(sequence_loss) / len(sequence_loss)
                    epoch_val_loss.append(sequence_loss)
                    print('{:<20} | FPS: {:5.2f} | mIoU: {:>5.2f} %'.format(seq_name, n_frames / sequence_time,
                                                                            sequence_iou * 100))

            epoch_val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
            epoch_val_iou = sum(epoch_val_iou) / len(epoch_val_iou)

            self.summary_writer.add_scalar('val_loss', epoch_val_loss, epoch + 1)
            self.summary_writer.add_scalar('val_iou', epoch_val_iou, epoch + 1)

            # Save best model (val_iou)
            if epoch_val_iou > self.best_val_iou:
                self.best_val_iou = epoch_val_iou
                save_model(self.model.state_dict(), self.args.job_name)

            print()
            print(f"End of Epoch {epoch + 1}/{self.args.num_epochs} | time: {int(time() - since_epoch) // 60}' | "
                  f"train loss: {epoch_train_loss:.4f} | val loss: {epoch_val_loss:.4f} | "
                  f"val mIoU: {epoch_val_iou * 100:.3f} %")
            print()

        print(f"Training completed. Elapsed time: {int(time() - since) // 60}' | Best validation mIoU: "
              f"{100*self.best_val_iou:.3f} %")
