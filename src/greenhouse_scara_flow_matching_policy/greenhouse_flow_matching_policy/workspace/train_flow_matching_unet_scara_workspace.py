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

    def save_checkpoint(self, *args, **kwargs):
        self._wait_for_checkpoint()
        return super().save_checkpoint(*args, **kwargs)

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
