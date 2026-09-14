# 3D Flow Matching (FM3)

Gewone conditional rectified flow op exact dezelfde DP3 PointNet en conditional
U-Net. `policy.py` erft de encoder, U-Net, normalisatie-interface en actie-extractie
van de meegeleverde upstream DP3. Alleen de trainingsloss en sampler veranderen.
Dit implementeert niet de afzonderlijke **FlowPolicy** consistency-methode.

- Training: `x_t = (1-t)*noise + t*action`, target-velocity `action - noise`.
- Inference: integreer het geleerde snelheidsveld van t=0 naar t=1.
- Euler kost één U-Net-evaluatie per stap; Heun twee.
- Standaard acht Heun-stappen (16 evaluaties), zonder garantie op voldoende taakprestatie.
- Dezelfde encoder-output 64, U-Net `[512,1024,2048]`, twee observaties, horizon 16,
  acht uitvoeracties en AdamW-betas als DP3. Dezelfde grenspadding in de loss.
- Dezelfde ongecropte geometrie, FPS naar 1.024 punten en train-only limits-normalisatie.

`config.yaml` gebruikt de echte single-camera-demo en bijbehorende intrinsics.
`config_dummy.yaml` gebruikt de twee synthetische camera's en expliciet fictieve
onderlinge pose. Beide profielen komen overeen met de DP3-profielen.
De oude adapter met de 2D-U-Net is bewaard als `legacy_policy.py` en `core/`;
nieuwe training gebruikt die niet. Oude adapter-checkpoints zijn geen checkpoints
voor deze nieuwe architectuur.

## DelftBlue-smoketest

Gebruik de bestaande diffusion-Pixi-omgeving met de DP3-dependencies; geen nieuwe
packages nodig. Synchroniseer eerst de bijgewerkte repository, en voer op DelftBlue uit:

```bash
cd ~/scara_ws/greenhouse-scara
export GREENHOUSE_PIXI_PROJECT="/scratch/$USER/thesis/envs/diffusion_policy"
export GREENHOUSE_FM3_DATASET_DIR="/scratch/$USER/thesis/datasets/greenhouse_dummy_dataset"
bash scripts/submit.sh fm3-smoke
```

`fm3-smoke` gebruikt de volledige architectuur, batch 1, één training- en
validatiestap en geen EMA, net als DP3-smoke. De normalisatie scant nog alle
trainingsframes. Logs melden de voortgang. Het script vraagt acht CPU's en
5 GB per CPU. Het checkpoint komt onder
`/scratch/$USER/thesis/runs/greenhouse_scara_3d_flow_matching_policy/job<JOBID>/latest.ckpt`.
`fm3` start normale training op het echte profiel. Voor een echte smoketest stel
je `GREENHOUSE_FM3_CONFIG=config.yaml` en de echte dataset in.

## Lokale benchmark

```bash
python scripts/fetch_checkpoints.py --fm3 /scratch/tomoyamartynow/thesis/runs/greenhouse_scara_3d_flow_matching_policy/job<JOBID>/latest.ckpt
```

Vervang `<JOBID>`. Zet daarna in `benchmark_inference.ipynb` de selectie op
`"fm3": REPO / "checkpoints/fm3/latest.ckpt"`. `FM3_STEPS` en `FM3_METHOD`
regelen de solver. DP3/FM3 delen dezelfde pointcloud-preprocessing, die buiten
het gemeten model-inferencegedeelte valt. [Benchmarkprotocol](../../docs/INFERENCE_COMPARISON.md).

## Verificatie

Tests vergelijken DP3/FM3-parameterstructuren, analytische Euler/Heun-integratie,
evaluatieaantallen, de velocity-target en checkpointtraining/herladen/hervatten.
CPU-tests verkleinen de U-Net; de standaard Slurm-configuratie blijft volledig.
Er is tijdens implementatie geen nieuwe GPU-job op DelftBlue ingediend.
