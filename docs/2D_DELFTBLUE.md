# 2D-training op DelftBlue

Deze zes experimenten gebruiken de demonstraties vanaf episode 11. Elk model
krijgt twee camerastandpunten en vijf actie- en toestandswaarden:
`[J1, J2, J3, J4, scissor_roll]`. RGBD gebruikt per camera een RGB-stream en
een depth-stream (afstand in meters gedeeld door 1 meter, plus validiteitsmasker).
RGB blijft op 480×640; RGBD wordt voor alle streams naar 240×320 verkleind om
het GPU-geheugen te beperken. Dit zijn dus afzonderlijke modellen en checkpoints.

| Model | RGB Slurm | RGBD Slurm | Offline inference |
| --- | --- | --- | --- |
| ACT | `scripts/slurm/act_rgb.slurm` | `scripts/slurm/act_rgbd.slurm` | `scripts/run/act_rgb.sh`, `scripts/run/act_rgbd.sh` |
| Diffusion Policy | `scripts/slurm/dp_rgb.slurm` | `scripts/slurm/dp_rgbd.slurm` | `scripts/run/dp_rgb.sh`, `scripts/run/dp_rgbd.sh` |
| Flow Matching | `scripts/slurm/fm_rgb.slurm` | `scripts/slurm/fm_rgbd.slurm` | `scripts/run/fm_rgb.sh`, `scripts/run/fm_rgbd.sh` |

De dataset staat volgens de opgegeven locatie in
`/scratch/tomoyamartynow/thesis/datasets/green_house_dataset1`. De jobs gebruiken
die map direct en selecteren alleen episodes met nummer 11 of hoger. Controleer
op DelftBlue dat dit pad bestaat:

```bash
cd ~/greenhouse-scara
ls /scratch/$USER/thesis/datasets
export GREENHOUSE_DATASET_DIR=/scratch/$USER/thesis/datasets/green_house_dataset1
ls "$GREENHOUSE_DATASET_DIR"/episode_11.hdf5
```

De job controleert vóór de training of minstens twee geselecteerde episodes
aanwezig en leesbaar zijn. Als de clusterbestanden symlinks zijn, moeten ook hun
doelen op DelftBlue aanwezig zijn. De dataset gaat niet mee via Git. Met
`GREENHOUSE_MIN_EPISODE` kun je de ondergrens expliciet wijzigen.

De jobs gebruiken de bestaande Pixi-projecten onder
`/scratch/$USER/thesis/envs/act` en
`/scratch/$USER/thesis/envs/diffusion_policy`. Die omgevingen en hun manifesten
staan niet in deze repository. Stel `GREENHOUSE_PIXI_PROJECT` in als jouw
`pixi.toml` ergens anders staat. De Slurm-account en GPU-partitie komen uit de
bestaande DelftBlue-jobs in dit project en zijn hier niet op de cluster getest.

Begin per variant met een korte proefrun, bijvoorbeeld:

```bash
ACT_NUM_EPOCHS=1 bash scripts/submit_2d.sh act rgb --time=00:30:00
GREENHOUSE_2D_EPOCHS=1 bash scripts/submit_2d.sh dp rgb --time=00:30:00
GREENHOUSE_2D_EPOCHS=1 bash scripts/submit_2d.sh fm rgb --time=00:30:00
```

Voor een volledige run:

```bash
bash scripts/submit_2d.sh act rgb
bash scripts/submit_2d.sh act rgbd
bash scripts/submit_2d.sh dp rgb
bash scripts/submit_2d.sh dp rgbd
bash scripts/submit_2d.sh fm rgb
bash scripts/submit_2d.sh fm rgbd
```

Elke job krijgt een eigen map onder
`/scratch/$USER/thesis/runs/2d/<model>_<input>_scissor_<jobid>`.
ACT bewaart onder meer `policy_best.ckpt`, `policy_last.ckpt`, `config.pkl` en
`dataset_stats.pkl`; DP en Flow bewaren `checkpoints/latest.ckpt`. Slurm-logbestanden
staan in de repository als `slurm-<jobid>.out`. De standaardtraining is 200 epochs;
de Slurm-limiet is 24 uur. Deze zes scripts hervatten een onderbroken run nog
niet automatisch; kies zo nodig een passende tijdslimiet vóór het indienen.
ACT gebruikt standaard pretrained ResNet18-gewichten; zet
`ACT_PRETRAINED_BACKBONE=0` als die niet in de Torch-cache op de compute node staan.
Batchgroottes kunnen via `ACT_BATCH_SIZE` of `GREENHOUSE_2D_BATCH_SIZE` omlaag.

## Inference op opnames

Deze scripts laden een checkpoint en voeren actievoorspelling uit op opgenomen
episodes. Ze bewegen de robot niet. Gebruik een omgeving met de bijbehorende
Python-dependencies. Voor ACT is het tweede argument het volledige pad naar
`policy_best.ckpt` of `policy_last.ckpt`; voor DP/Flow naar `latest.ckpt`.

```bash
bash scripts/run/act_rgb.sh /pad/naar/act_rgb_run/policy_best.ckpt "$GREENHOUSE_DATASET_DIR"
bash scripts/run/dp_rgb.sh /pad/naar/dp_rgb_run/checkpoints/latest.ckpt "$GREENHOUSE_DATASET_DIR"
bash scripts/run/fm_rgbd.sh /pad/naar/fm_rgbd_run/checkpoints/latest.ckpt "$GREENHOUSE_DATASET_DIR"
```

Vervang `rgb` door `rgbd` waar nodig. ACT toont de MSE per actie; DP en Flow
schrijven hun evaluatie naar `runs/offline_eval/`. Met
`GREENHOUSE_INFERENCE_DEVICE=cuda:0` gebruik je een GPU. Gebruik daarvoor een
compute node; standaard wordt CPU gebruikt. Het opgegeven datasetpad vervangt
ook het pad dat in een DP/Flow-checkpoint is opgeslagen.
De scripts controleren of het checkpoint bij RGB of RGBD en vijf dimensies hoort;
bestaande checkpoints met vier robotjoints worden geweigerd.
Voor een snelle controle kun je `GREENHOUSE_EVAL_SAMPLES=1` en voor DP/Flow
`GREENHOUSE_INFERENCE_STEPS=1` instellen; dat is geen kwaliteitsmeting.

De bestaande live-runners ondersteunen al twee RealSense-camera's via
`--camera NAME SERIAL`, maar checkpoints met vijf dimensies worden daar bewust
geweigerd: de gemeten schaarstand en het commando naar de fysieke schaar hebben
nog geen live-adapter. De offline scripts hierboven werken wel met die vijfde
dimensie. Voor live uitvoering moeten de schaarinterface, grenzen en veilige
bewegingsvolgorde eerst aan de robot worden gekoppeld.
