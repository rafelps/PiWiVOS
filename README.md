# Fast Video Object Segmentation by Pixel-Wise Feature Comparison (PiWiVOS)

Code for my final degree thesis [Fast Video Object Segmentation by Pixel-Wise Feature Comparison (PiWiVOS)][thesis].

This final degree thesis tackles the task of [One-Shot Video Object Segmentation][davis], where multiple objects have to
be separated from the background using the ground truth masks for them in the very first frame only. Objects' large pose
and scale variations throughout the sequence, alongside occlusions happening among them, make this task extremely
challenging. Fast Video Object Segmentation by Pixel-Wise Feature Comparison&mdash;which is trained and tested on the
well-known [DAVIS dataset][davis]&mdash;goes a step further, and besides achieving comparable results with
state-of-the-art methods, it works one order of magnitude faster than them, or even two in some cases.

## Dependencies

This version of the project (updated from the original one for better reproducibility) has been build using:

- [Python][python] 3.7
- [PyTorch][pytorch] 1.8.1 + Torchvision 0.9.1
- [NumPy][numpy] 1.19.5
- [Pillow][pil] 8.3.0
- [Scikit-image][scikit-image] 0.16.2

## Usage

There are two main scripts: `train.py`, which serves to train a model; and `test.py`, which is used to evaluate a
checkpoint and optionally export the predicted masks.

### `train.py`

The complete usage can be seen typing:

```
$ train.py -h
```

This script has many arguments which control specific parts of our method. Section 4.2 of the [Thesis][thesis]
introduces all these parameters, and Chapter 5 presents a complete study of their optimal values, in which they are set
by default.

Apart from method-specific parameters, the most important arguments are:

- `--job_name JOB_NAME`: Used to identify the job and create a log directory for it at `logs/JOB_NAME`, in which
  tensorboard logs and checkpoints will be stored.

- `--path PATH`: Path to the [DAVIS dataset][davis]. Defaults to `data/DAVIS`.
- `--model_name ['piwivos', 'piwivosf']`: Name of the model to use. PiWiVOS uses a resnet50 backbone while PiWiVOS-F
  uses a resnet34 and has lower output resolution. See Chapter 5 of the [Thesis][thesis] for more information. Defaults to
  `'piwivos'`.

The script trains the model from a pre-trained ResNet using the official DAVIS 2017 `train` set, and validates using
the `val` one.

### `test.py`

The complete usage can be seen typing:

```
$ test.py -h
```

The main arguments are:

- `--path PATH`: Path to the [DAVIS dataset][davis]. Defaults to `data/DAVIS`.
- `--checkpoint_path CHECKPOINT_PATH`: Path to the checkpoint file (.pth) to evaluate. Defaults to
  `checkpoints/piwivos/piwivos.pth` following this repository's structure.

- `--model_name ['piwivos', 'piwivosf']`: Name of the model to use. Must match with the loaded checkpoint. Defaults
  to `'piwivos'`.

- `--image_set ['val', 'test-dev', 'test-challenge']`: Set of images on which to evaluate the model. Defaults to `'val'`.

- `--export`: When set, the script exports the predicted masks in the disk. These are stored in a `results`
  subdirectory side by side the evaluated checkpoint.

### Data

PiWiVOS is trained and evaluated using the [DAVIS][davis] 2017 semi-supervised 480p dataset, which can be downloaded
from [this link][davisd].

Nonetheless, our code can also be used with different DAVIS data. In first place, our dataloader supports the 
DAVIS 2016 semi-supervised 480p dataset, which is a subset of the DAVIS 2017 version and contains 
single-object sequences, being an easier task. However, if the user wants to perform this task on the (larger) DAVIS 
2017 dataset, the dataloader has also an option to merge individual object masks into a "single-object mask".

See the [DAVIS dataloader][dataloader].

## Results

Results reported by this repository's checkpoints are slightly better than the ones in the [Thesis][thesis] 
strictly due to seeding and possible library updates.

| Model Name | J Mean | F Mean | G Mean (J&F)|
| --- | :---: | :---: | :---: |
|PiWiVOS|67.95%|74.93%|71.42%|
|PiWiVOS-F|56.17%|54.46%|55.32%|

<!---|PiWiVOS (single-object)|76.71%|78.55%|77.63%|--->
Table: Results on the `val` set. See [Thesis][thesis] for the original results on `val` and `test-dev` sets. 




## Citation
You can cite our work using:

```bibtex
@phdthesis{Palliser Sans_2019,
	title={Fast Video Object Segmentation by Pixel-Wise Feature Comparison},
	url={http://hdl.handle.net/2117/169370},
	abstractNote={This final degree thesis tackles the task of One-Shot Video Object Segmentation, where multiple objects have to be separated from the background only 			having the ground truth masks for them in the very first frame. Their large pose and scale variations throughout the sequence, and the occlusions happening 			between them make this task very difficult to solve. Fast Video Object Segmentation by Pixel-Wise Feature Comparison goes a step further, and besides achieving 		comparable results with state-of-the-art methods, it works one order of magnitude faster than them, or even two in some cases.},
	school={UPC, Centre de Formació Interdisciplinària Superior, Departament de Teoria del Senyal i Comunicacions},
	author={Palliser Sans, Rafel},
	year={2019},
	month={May},
}
```

[python]: https://www.python.org/

[pytorch]: https://pytorch.org/

[numpy]: https://numpy.org/

[pil]: https://pillow.readthedocs.io/en/latest/index.html

[scikit-image]: https://scikit-image.org/

[thesis]: http://hdl.handle.net/2117/169370

[davis]: https://davischallenge.org/

[davisd]: https://davischallenge.org/davis2017/code.html

[dataloader]: dataloaders/davis.py
