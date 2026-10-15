from pathlib import Path
import os

from greenhouse_flow_matching_policy.workspace.train_flow_matching_unet_image_workspace import (
    TrainFlowMatchingUnetImageWorkspace,
)


class TrainFlowMatchingUnetScaraWorkspace(TrainFlowMatchingUnetImageWorkspace):
    """SCARA flow-matching training with serialized final checkpoint writes."""

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

        last_epoch = self.epoch - 1
        final_epoch_was_saved = (
            last_epoch >= 0
            and last_epoch % self.cfg.training.checkpoint_every == 0
        )
        if self.cfg.checkpoint.save_last_ckpt and not final_epoch_was_saved:
            self.save_checkpoint()

        self._wait_for_checkpoint()
