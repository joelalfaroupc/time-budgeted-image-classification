# Artifact provenance and validation

## Original artifacts

The submitted Python source and supplied checkpoint are preserved unchanged. The source was provided as pasted text; the checkpoint was supplied separately. No dataset archive or class-name mapping for the final 12-class model was provided.

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| `archive/original_training.py` | `cde96d9f8cf362b7b8c6d84420ab04f9f27e34c2ae7487e26228ab4fb9bd1d30` | 6234 |
| `checkpoints/best_model.pth` | `71f07ad1ecdbfc2dbf173c6a89975045b7c3fdcfb1297fee9416ced3223d8b5d` | 1845173 |

The original `ElVostreModel` definition was checked against `model.py`. The supplied state dictionary loads strictly, has a 12-output final layer, and yields finite scores on a shape-compatibility check with two 28×28 grayscale inputs. This input size is a verification fixture, not evidence of the competition's original resolution.

## Historical performance

`results/validation_history.csv` and `assets/validation-history.svg` transcribe the instructor's feedback: seven validation results, with a maximum of 78.26% at epoch 4. The feedback states that the time limit was reached before epoch-7 validation. The private correspondence image is not redistributed. No exact hidden-test score, grade or ranking is asserted here.

The supplied checkpoint filename does not prove that its weights correspond to the best recorded epoch. In the original loop, `best_model_state = model.state_dict()` retained references to live tensors. Later optimization could therefore overwrite the purported best snapshot. The original source also contains no checkpoint-save call. Its relationship to the separately supplied file cannot be reconstructed conclusively.

## Changes in the documented edition

- Extract the original architecture into `model.py`, keeping all state-dictionary names and dimensions compatible.
- Save detached, cloned weights immediately when validation accuracy improves.
- Drive `ReduceLROnPlateau` with validation loss; the original used the last training batch's loss.
- Weight logged mean losses by the number of images; the original validation loss averaged batch means.
- Validate train/validation class names and order, and persist class mapping for new checkpoints.
- Remove the unused 1,000-image subset; it otherwise prevented execution on smaller datasets.
- Add explicit paths, run configuration, a monotonic time check and inference commands.
- Allow an optional resize and disabling horizontal flips for other datasets; the defaults keep the original preprocessing.
- Reject nonpositive training budgets and existing output directories.

The corrections were checked on generated images. No original-dataset retraining was performed, so the historical metrics are not represented as reproduced results. The original source remains archival and is not the recommended entry point. Running it would additionally require its original `matplotlib` and `tqdm` imports.

## Dataset identity

The supplied weights have 12 output classes, but their names are unknown. Course starter files containing EMNIST-related assets are not evidence that this final checkpoint classifies those starter categories, and have not been included. Predictions from the original checkpoint expose anonymous class indices until the correct original mapping is recovered.
