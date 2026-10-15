from pathlib import Path
import os

from greenhouse_diffusion_policy.workspace.train_diffusion_unet_image_workspace import (
    TrainDiffusionUnetImageWorkspace,
)


class TrainDiffusionUnetScaraWorkspace(TrainDiffusionUnetImageWorkspace):
    """SCARA training with serialized and final checkpoint writes.

    The upstream workspace writes checkpoints on background threads. When a
    latest and top-k checkpoint are requested together, complete CPU copies of
    the model, EMA model, and optimizer can coexist. Waiting for an outstanding
    write bounds that peak while retaining upstream checkpoint compatibility.
    """

    def _wait_for_checkpoint(self) -> None:
        thread = self._saving_thread
        if thread is not None and thread.is_alive():
            thread.join()
        self._saving_thread = None

    def save_checkpoint(self, path=None, tag='latest', exclude_keys=None,
                        include_keys=None, use_thread=True):
        # Complete a temporary file before replacing the previous checkpoint.
        # Synchronous writes also propagate disk errors to the training job.
        self._wait_for_checkpoint()
        destination = Path(path) if path is not None else self.get_checkpoint_path(tag)
        temporary = destination.with_name(destination.name + '.tmp')
        super().save_checkpoint(path=temporary, tag=tag,
                                exclude_keys=exclude_keys, include_keys=include_keys,
                                use_thread=False)
        os.replace(temporary, destination)
        return str(destination.absolute())

    def run(self) -> None:
        super().run()

        # Upstream checks before incrementing the epoch. Unless the final epoch
        # happens to align with checkpoint_every, save the actual final state.
        last_epoch = self.epoch - 1
        final_epoch_was_saved = (
            last_epoch >= 0
            and last_epoch % self.cfg.training.checkpoint_every == 0
        )
        if self.cfg.checkpoint.save_last_ckpt and not final_epoch_was_saved:
            self.save_checkpoint()

        # Never let the Slurm process exit while a checkpoint is still writing.
        self._wait_for_checkpoint()
