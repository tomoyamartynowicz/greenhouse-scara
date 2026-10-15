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
in een DelftBlue SSH-loginterminal dat dit pad bestaat. Een lokale terminal in
een via SFTP/GVFS geopende map heeft geen `sbatch`:

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
Elke 2D-job vraagt vier CPU's en `--mem-per-cpu=5G`, gelijk aan de bestaande
ACT/DP/Flow-Slurm-scripts. DelftBlue accepteerde de eerdere `--mem=32G`-variant
niet bij het indienen.

Je kunt elk Slurm-script rechtstreeks indienen vanuit de repositoryroot. De
eerdere ACT RGB-proefrun (job 994975) slaagde. Voor de andere varianten kun je
desgewenst een korte proefrun doen:

```bash
GREENHOUSE_2D_EPOCHS=1 sbatch --export=ALL --time=00:30:00 scripts/slurm/dp_rgb.slurm
GREENHOUSE_2D_EPOCHS=1 sbatch --export=ALL --time=00:30:00 scripts/slurm/fm_rgb.slurm
```

Voor een volledige run:

```bash
sbatch --export=ALL scripts/slurm/act_rgb.slurm
sbatch --export=ALL scripts/slurm/act_rgbd.slurm
sbatch --export=ALL scripts/slurm/dp_rgb.slurm
sbatch --export=ALL scripts/slurm/dp_rgbd.slurm
sbatch --export=ALL scripts/slurm/fm_rgb.slurm
sbatch --export=ALL scripts/slurm/fm_rgbd.slurm
```

`scripts/submit_2d.sh act rgb` is slechts een snelkoppeling naar
`sbatch scripts/slurm/act_rgb.slurm`. Het Slurm-bestand bevat de resource-aanvraag
en start `scripts/train_2d_job.sh`, waarin het Pixi/Python-trainingscommando staat.

Elke job krijgt een eigen map onder
`/scratch/$USER/thesis/runs/2d/<model>_<input>_scissor_<jobid>`.
ACT bewaart `policy_best.ckpt` bij een verbeterde validatieloss en
`policy_last.ckpt` na de eerste epoch en daarna elke 10 epochs. Elke 100 epochs
komt er een genummerd checkpoint; alleen de vijf nieuwste blijven bewaard.
Daarnaast blijven `config.pkl` en `dataset_stats.pkl` beschikbaar. DP en Flow
bewaren na elke epoch `checkpoints/latest.ckpt` plus maximaal drie checkpoints
met de laagste validatieloss. Voltooide checkpointbestanden worden pas vervangen
nadat het nieuwe bestand volledig is geschreven. Een achtergebleven `.tmp` na
een afgebroken job is geen bruikbaar checkpoint.

Slurm-logbestanden staan in de repository als `slurm-<jobid>.out`.
Alle zes jobs hebben een tijdslimiet van 10 uur en een hoge bovengrens van
100000 epochs, zodat de tijdslimiet normaal de run beëindigt. ACT gebruikt
chunk 30, learning rate `5e-5` en KL-weight 10. DP/Flow gebruiken learning rate
`1e-4` na 500 warmup-stappen (`constant_with_warmup`); de learning rate hangt
hierdoor niet af van de kunstmatig hoge epochlimiet.
Alle zes gebruiken standaard batchgrootte 64 direct,
zonder gradientaccumulatie. Met `ACT_BATCH_SIZE` of
`GREENHOUSE_2D_BATCH_SIZE` kun je die grootte per model wijzigen als een
variant niet in het GPU-geheugen past. De succesvolle ACT-proefrun had batch 2;
batch 64 is voor deze tweecamera-varianten nog niet op DelftBlue getest.
ACT gebruikt standaard geen
pretrained ResNet18, net als de geslaagde proefrun; zet
`ACT_PRETRAINED_BACKBONE=1` als je de gewichten in de Torch-cache hebt.

Slurm kan na 10 uur een job als `TIMEOUT` beëindigen. Gebruik dan het laatst
volledig geschreven checkpoint; werk sinds dat checkpoint gaat verloren.
De scripts hervatten een onderbroken run nog niet automatisch. ACT-checkpoints
bevatten modelgewichten, geen optimizerstatus voor exact hervatten.
Bij ACT wordt `policy_best.ckpt` aan het begin van elke epoch beoordeeld; bij
een enkele epoch is dat dus nog het ongetrainde model. Gebruik voor zo'n
proefrun `policy_last.ckpt`.

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
