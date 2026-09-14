# Greenhouse SCARA policies

Eén repository voor de drie 2D-policies, DP3 en hun gedeelde HDF5/preprocessing-code.

Nieuw: [DP3 zonder pointcloud-crop](src/greenhouse_scara_3d_diffusion_policy/README.md)
en [interactieve controle van je echte RGB-D-demo](inspect_pointcloud.ipynb).

```text
src/
  greenhouse_scara_common/
  greenhouse_scara_act/
  greenhouse_scara_diffusion_policy/
  greenhouse_scara_flow_matching_policy/
  greenhouse_scara_3d_common/
  greenhouse_scara_3d_diffusion_policy/
calibration/
environments/requirements-train.txt
scripts/submit.sh
scripts/slurm_env.sh
```

Houd deze bronmappen samen. De train/eval-entrypoints voegen de gedeelde
`src`-map zelf aan het importpad toe. De originele `act`, `diffusion_policy`,
ROS-workspace, recorder en 3D-projecten zijn niet nodig voor deze trainingsroutes.
Modelarchitecturen, losses, inputconfiguraties en checkpoint-modulepaden zijn bij
de verhuizing behouden. Bestaande checkpoints kunnen nog oude absolute data- of
runpaden bevatten; controleer die bij hervatten of offline evaluatie.

## Git

Maak alleen deze map een repository, niet de volledige `scara_ws`:

```bash
cd /home/tomoya/scara_ws/greenhouse-scara
git init -b main
git add .
git status
git commit -m "Add greenhouse SCARA training projects"
git remote add origin <repository-url>
git push -u origin main
```

Datasets, checkpoints, runs, caches en omgevingsinstallaties worden genegeerd.
YAML-configuraties, Slurm-scripts, README's en de oorspronkelijke licenties gaan
wel mee. Er is nog geen remote aangemaakt en niets gepubliceerd.

Op DelftBlue kun je bijvoorbeeld clonen naar `~/greenhouse-scara`. Iedere andere
clone-locatie werkt ook. De oorspronkelijke lokale mappen onder `scara_ws/src`
zijn symlinks naar deze bronmappen, zodat bestaande commando's en de lokale
3D-code hun gedeelde imports behouden. Die symlinks staan buiten deze repository
en zijn niet nodig op DelftBlue.

## Omgeving

Gebruik Python 3.10 en een passende GPU-versie van PyTorch/torchvision.
`environments/requirements-train.txt` legt de directe trainingsdependencies vast
zoals geïnstalleerd in de lokaal geteste `act`-omgeving. Het is geen volledige
lockfile en een nieuwe installatie op DelftBlue is nog niet getest.

De lokale Torch-build is CUDA 12.1. Bij een overeenkomstige omgeving:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.3.1+cu121' 'torchvision==0.18.1+cu121'
python -m pip install -r environments/requirements-train.txt
```

De Slurm-jobs gebruiken de bestaande Pixi-omgeving op de cluster. Standaard zoeken
ze voor ACT in `/scratch/$USER/thesis/envs/act` en voor diffusion én flow
in `/scratch/$USER/thesis/envs/diffusion_policy`. Met `GREENHOUSE_PIXI_PROJECT` kies je zelf een map met
`pixi.toml`. Deze bestaande clusteromgevingen zijn niet meegekopieerd: hun
manifesten waren lokaal niet beschikbaar. Zet een gevalideerde `pixi.toml` en
`pixi.lock` onder `environments/` zodra die beschikbaar zijn.

Voor training zijn robot, ROS en RealSense niet nodig. Live-evaluatie heeft wel
hardware en `pyrealsense2` nodig. De optionele Ray-tools vereisen daarnaast Ray.
ACT gebruikt standaard pretrained ResNet18-gewichten; zorg dat die in
`TORCH_HOME` beschikbaar zijn, of gebruik `--no-pretrained_backbone` voor een test.

## Dataset en lokale training

De dataset blijft buiten Git. Geef een absoluut pad op. Voor lokale training:

```bash
export GREENHOUSE_DATASET_DIR=/home/tomoya/scara_ws/datasets/greenhouse_dummy_dataset

python src/greenhouse_scara_act/imitate_episodes.py \
  --dataset_dir "$GREENHOUSE_DATASET_DIR" \
  --ckpt_dir "$HOME/greenhouse_runs/act/run01"

python src/greenhouse_scara_diffusion_policy/train.py \
  --config-name=train_diffusion_unet_scara_workspace

python src/greenhouse_scara_flow_matching_policy/train.py \
  --config-name=train_flow_matching_unet_scara_workspace
```

Diffusion/flow vereisen `GREENHOUSE_DATASET_DIR` of een expliciete override
`task.dataset_path=/absoluut/pad`. Hun standaard uitvoermappen staan onder
`$HOME/greenhouse_runs/diffusion_policy` en `flow_matching_policy`, onafhankelijk
van de werkdirectory. `GREENHOUSE_DP_RUN_ROOT` en `GREENHOUSE_FM_RUN_ROOT`
overschrijven deze defaults.

**Bij direct starten met Python blijft `inputs.yaml` top-RGB + top-depth.**
De ACT-Slurm-job selecteert expliciet `inputs_rgb_dual.yaml`: top-RGB + bottom-RGB,
zonder depth. Met `GREENHOUSE_ACT_INPUTS_CONFIG` kies je een ander absoluut YAML-pad. Voor de besproken twee-camera
RGB-baseline vervang je `camera_top_depth` door `camera_bottom_rgb`, met
`camera: camera_bottom`, `type: rgb` en dezelfde shape als top-RGB, in ACT's
`inputs.yaml` en diffusion/flow's `config/task/scara_image.yaml`. Dit is een
aparte modelconfiguratiewijziging; de verhuizing schakelt geen modaliteiten om.

## DelftBlue

Maak in de online shell eerst de datasetmap:

```bash
mkdir -p /scratch/$USER/thesis/datasets/greenhouse_dummy_dataset
```

Kopieer vanaf de lokale computer (vervang `NETID`):

```bash
rsync -avP /home/tomoya/scara_ws/datasets/greenhouse_dummy_dataset/ \
  NETID@login.delftblue.tudelft.nl:/scratch/NETID/thesis/datasets/greenhouse_dummy_dataset/
```

Dien vanuit de cluster-shell jobs in via de wrapper. Die werkt ook buiten de
repositorymap en bewaart het echte repositorypad wanneer Slurm het script kopieert:

```bash
export GREENHOUSE_DATASET_DIR=/scratch/$USER/thesis/datasets/greenhouse_dummy_dataset
# Optioneel: één bestaande Pixi-omgeving voor alle drie.
# export GREENHOUSE_PIXI_PROJECT=/pad/naar/map/met/pixi.toml

bash ~/greenhouse-scara/scripts/submit.sh act
bash ~/greenhouse-scara/scripts/submit.sh dp
bash ~/greenhouse-scara/scripts/submit.sh fm
```

Controleer account, partitie, tijd, batchgrootte en geheugen in de Slurm-scripts.
ACT behoudt zijn bestaande sweep van zes jobs; het is geen korte proefrun.
Je kunt sbatch-opties meegeven, bijvoorbeeld `submit.sh act --array=0`.
Rechtstreeks `sbatch src/.../train_*.slurm` werkt vanuit de repositoryroot, of met
`GREENHOUSE_REPO_DIR` expliciet ingesteld. Gebruik de normale overerving van
omgevingsvariabelen; `--export=NONE` verwijdert ook de ingestelde paden.

Clusterdefaults: datasets onder `/scratch/$USER/thesis/datasets`, runs onder
`/scratch/$USER/thesis/runs`, Torch-cache onder `/scratch/$USER/thesis/cache/torch`.
Overrides: `GREENHOUSE_DATASET_DIR`, `GREENHOUSE_RUN_ROOT`,
`GREENHOUSE_ACT_RUN_ROOT`, `GREENHOUSE_DP_RUN_ROOT`, `GREENHOUSE_FM_RUN_ROOT`,
`GREENHOUSE_PIXI_PROJECT`, `TORCH_HOME`.

Bewaar bij experimenten de Git-commit en configuratie. Werk een checkout niet bij
terwijl een job daar nog code uit moet laden. Haal belangrijke resultaten van
scratch terug naar permanente opslag.

## Controles

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s src/greenhouse_scara_common/tests -v
```

De bestaande tests omvatten ook live-camera-mocks en gebruiken daarvoor
`pyrealsense2`. Pure trainingsruns importeren die hardwaremodule niet.

Meer modelspecifieke informatie staat in de README van iedere bronmap; gebruik
de paden en clusterinstructies hierboven voor deze repository-indeling.

## ACT: eerste GPU-test en sweep

Na het bijwerken van de clone op DelftBlue:

```bash
export GREENHOUSE_DATASET_DIR=/scratch/$USER/thesis/datasets/greenhouse_dummy_dataset
export GREENHOUSE_PIXI_PROJECT=/scratch/$USER/thesis/envs/act
bash ~/greenhouse-scara/scripts/submit.sh act-smoke
```

`act-smoke` dient één arraytaak in met maximaal 30 minuten: batch 2, één epoch,
één train- en één validatiestap, en nul dataloaderworkers. De normale ACT-
architectuur en beide RGB-inputs op 480×640 blijven behouden. CUDA is verplicht;
een niet werkende GPU-configuratie valt niet stilzwijgend terug op CPU.
Pretrained backbonegewichten blijven standaard ingeschakeld. Als deze niet in
`TORCH_HOME` staan en je zonder download wilt testen, zet voor de proef
`export ACT_PRETRAINED_BACKBONE=0`; verwijder dit met `unset ACT_PRETRAINED_BACKBONE`
vóór training met pretrained gewichten. Dit is een technische test, geen benchmark.

Vervolgens start `bash ~/greenhouse-scara/scripts/submit.sh act` de zes sweepjobs:
batches `[64,64,64,128,128,128]`, chunks `[15,30,45,15,30,45]` en learning rates
`[5e-5,5e-5,5e-5,1e-4,1e-4,1e-4]`, met 5000 epochs en KL-weight 10.
Overrides zijn `ACT_BATCH_SIZE`, `ACT_CHUNK_SIZE`, `ACT_LR`, `ACT_NUM_EPOCHS`
en `ACT_KL_WEIGHT`; smoke-modus legt batch 2 en één epoch vast.

Let op: ACT kiest één willekeurig tijdstip per demo per epoch. De 20 demo's
worden 16 train- en 4 validatiedemo's. Batch 64 en 128 geven daardoor beide
slechts 16 trainingssamples per batch. Gebruik bijvoorbeeld batch 2, 4 of 8 voor
een batchvergelijking op deze kleine dataset.

Vroege foutmeldingen staan in `slurm-<arrayjob>_<taak>.out` in de repositoryroot.
Trainingslogs en checkpoints staan in
`/scratch/$USER/thesis/runs/greenhouse_scara_act/{smoke,chunk_sweep}/<runnaam>/`.
Iedere runnaam bevat het jobnummer, zodat herhaalde tests geen checkpoints van
een eerdere run overschrijven.

## Diffusion: kort checkpoint voor inference-metingen

```bash
export GREENHOUSE_DATASET_DIR=/scratch/$USER/thesis/datasets/greenhouse_dummy_dataset
export GREENHOUSE_PIXI_PROJECT=/scratch/$USER/thesis/envs/diffusion_policy
bash ~/greenhouse-scara/scripts/submit.sh dp-smoke
```

Gebruik het werkelijke clone-pad (bijvoorbeeld `~/scara_ws/greenhouse-scara`).
`dp-smoke` gebruikt `train_diffusion_unet_scara_smoke.yaml`: beide RGB-camera's op
480×640, qpos, de huidige volledige U-Net `[256,512,1024]`, horizon 16,
2 observaties, 8 uitvoeracties en 100 DDIM-inference-stappen. Eén epoch bevat één
train- en één validatiestap met batch 1. EMA staat uit voor deze technische test;
het checkpoint bevat de gewone policy. De offline evaluatie genereert ook acties.

Na succes staat het bestand onder:
`/scratch/$USER/thesis/runs/greenhouse_scara_diffusion_policy/train_diffusion_unet_scara_smoke_job<JOBID>/checkpoints/latest.ckpt`.
De uitvoer staat in `slurm-<JOBID>.out` in de repositoryroot.
`GREENHOUSE_DP_CONFIG` en `GREENHOUSE_DP_RUN_NAME` kunnen config en runnaam
overschrijven bij de normale `dp`-opdracht; `dp-smoke` kiest zijn eigen config.
De normale `dp`-config blijft top-RGB + top-depth.

Een kort getraind checkpoint volstaat voor technische latency-metingen, niet
voor policykwaliteit of robotbesturing. Houd tijdens vergelijken hardware,
batchgrootte en gemeten scope gelijk; vermeld resolutie, observatiehistorie,
actiechunk en denoising-stappen. Warm het model op en synchroniseer CUDA rond
de meting. Het live-script gebruikt standaard 16 inference-stappen en overschrijft
dus de 100 uit de trainingsconfig; kies het aantal expliciet bij de benchmark.
ACT en de diffusion RGB-smoke-config gebruiken beide 480×640 zonder crop.
Diffusion gebruikt wel twee observatietijdstappen per camera, ACT één.
De smoke-config verkleint de netwerkarchitectuur niet.

## Flow matching: dezelfde RGB-input als diffusion

```bash
cd ~/scara_ws/greenhouse-scara
export GREENHOUSE_DATASET_DIR=/scratch/$USER/thesis/datasets/greenhouse_dummy_dataset
export GREENHOUSE_PIXI_PROJECT=/scratch/$USER/thesis/envs/diffusion_policy
bash scripts/submit.sh fm-smoke
```

Werk eerst de cluster-clone bij. `fm-smoke` kiest de nieuwe
`train_flow_matching_unet_scara_smoke.yaml`: top-RGB + bottom-RGB op 480×640,
geen depth of crops, twee observatietijdstappen, horizon 16 en acht uitvoeracties.
De encoder en volledige U-Net-grootte zijn gelijk aan de huidige diffusion-test.
Batch 1, één train- en validatiestap, geen EMA en maximaal 30 minuten.

Flow-loss en sampler blijven ongewijzigd: standaard 16 Heun-stappen, met twee
U-Net-evaluaties per stap (32 totaal). Diffusion gebruikt standaard 100 DDIM-
stappen. Vermeld bij een snelheidsvergelijking dus ook de sampler en het aantal
netwerkevaluaties; gelijke aantallen stappen zijn niet automatisch evenveel werk.

Het checkpoint staat onder
`/scratch/$USER/thesis/runs/greenhouse_scara_flow_matching_policy/train_flow_matching_unet_scara_smoke_job<JOBID>/checkpoints/latest.ckpt`.
De Slurm-log staat in `slurm-<JOBID>.out` in de repositoryroot.
`GREENHOUSE_FM_CONFIG` en `GREENHOUSE_FM_RUN_NAME` bieden config-/runnaamoverrides
voor de normale `fm`-opdracht. De gewone RGB-D-config blijft ongewijzigd.

## Eenvoudige inference-benchmark op je laptop

Gebruik bij voorkeur `benchmark_inference.ipynb`: checkpointpaden invullen en de
cellen uitvoeren. ACT, DP en FM staan klaar; DP3 is optioneel (`None` overslaan).
De Python-helper bevat de gedeelde meetcode. Pointcloud-opbouw/FPS, beeldverwerking
en transfers vallen buiten de meting; dit is model-inference, geen volledige robotlus.

Download op je **laptop** de gewenste checkpoints met SSH/rsync. Geef absolute
remote **bestandspaden**, geen runmappen; vervang hieronder de voorbeeldpaden:

```bash
python scripts/fetch_checkpoints.py \
  --act /scratch/tomoyamartynow/thesis/runs/ACT_RUN/policy_last.ckpt \
  --dp /scratch/tomoyamartynow/thesis/runs/DP_RUN/checkpoints/latest.ckpt \
  --fm /scratch/tomoyamartynow/thesis/runs/FM_RUN/checkpoints/latest.ckpt
```

Je kunt modellen weglaten of `--dp3 /scratch/.../job123/latest.ckpt` toevoegen.
Standaardhost: `tomoyamartynow@login.delftblue.tudelft.nl`; wijzig met `--host`.
`--dry-run` toont alleen de downloadcommando's, zonder verbinding.
Alle bestanden komen in `checkpoints/<model>/`, direct passend bij de notebook.
ACT krijgt automatisch `config.pkl` en `dataset_stats.pkl` mee. Downloads vervangen
de lokale selectie voor dat model pas nadat alle vereiste bestanden ontvangen zijn.
Gebruik afgeronde runs. De lokale dataset wordt niet opnieuw gedownload.

`scripts/benchmark_inference.py` laadt de modellen één voor één en meet batch-1
inference op enkele observaties uit één HDF5-episode. Geen camera/robot nodig.
Gebruik de gezamenlijke modelomgeving (lokaal werkte de `act`-Conda-omgeving).

Kopieer van DelftBlue:

- ACT: `policy_last.ckpt` (of `policy_best.ckpt`), **plus** `config.pkl` en
  `dataset_stats.pkl` in dezelfde lokale map.
- Diffusion en flow: ieder hun `checkpoints/latest.ckpt`; die bevatten hun
  eigen configuratie en normalisatie. Houd ze in afzonderlijke mappen.

Bijvoorbeeld vanuit de repositoryroot, na het kopiëren naar `checkpoints/`:

```bash
python scripts/benchmark_inference.py \
  --act checkpoints/act/policy_last.ckpt \
  --dp checkpoints/dp/latest.ckpt \
  --fm checkpoints/fm/latest.ckpt \
  --dataset /home/tomoya/scara_ws/datasets/greenhouse_dummy_dataset \
  --device cuda --warmup 3 --iterations 20 \
  --output runs/inference_speed.json
```

Je mag ook maar één checkpoint opgeven. `--device auto` (default) gebruikt CUDA
als beschikbaar, anders CPU. `--device cpu --threads 4` meet expliciet met vier
CPU-threads. `--episode 0 --states 3` kiest drie huidige frames verspreid over
episode 0; iedere policy krijgt daarbij zijn eigen vereiste observatiehistorie.
De shapes en normalisatie komen uit het checkpoint; cluster-datasetpaden worden
niet gebruikt en er worden geen pretrained gewichten gedownload.

Standaard blijven de inference-stappen uit het checkpoint behouden. Gebruik
bijvoorbeeld `--dp-steps 16 --fm-steps 16` om die expliciet te veranderen.
Bij Heun zijn 16 stappen 32 U-Net-evaluaties. Sampler, stappen, beeldshapes,
historie en actiechunk worden bij de resultaten vermeld.

De meting omvat policy-inference inclusief tensor-normalisatie en terugschaalbare
acties. Beelden/qpos worden vooraf verwerkt en op het gekozen device gezet.
Schijf-I/O, resizing, CPU/GPU-transfers, cameralezen en robotcommunicatie tellen
niet mee. CUDA wordt gesynchroniseerd en warm-upcalls worden niet meegerekend.
Er is geen frame-ratebegrenzer. Uitvoer: gemiddelde, mediaan, p95 in ms en calls/s.
Een call voorspelt een volledige actiechunk; calls/s is geen robotbesturings-Hz
of bewijs dat het model goede acties produceert. Een kort getraind checkpoint
is hiervoor voldoende. De optionele JSON bewaart de meetresultaten.

### Notebook

Open `benchmark_inference.ipynb` in de repositoryroot en selecteer de lokale
`act`-Pythonkernel (met PyTorch, pandas en de modeldependencies). Vul de
checkpointpaden in de eerste codecel in; zet een ontbrekend model op `None`.
Voer daarna de setup- en meetcel uit. De uitvoer toont een vergelijkingstabel;
de laatste cel bewaart optioneel JSON/CSV met een unieke tijdstempel.
Het notebook gebruikt dezelfde `scripts/benchmark_inference.py`, zodat de
timing en checkpointloaders gelijk blijven. Er is geen robotverbinding nodig.

De flow-job gebruikt standaard de bestaande **diffusion_policy**-Pixi-omgeving,
want dezelfde dependencies volstaan voor beide modellen. Als een eerder
geëxporteerde `GREENHOUSE_PIXI_PROJECT` nog naar de niet bestaande
`envs/flow_matching_policy` wijst, zet deze expliciet op
`/scratch/$USER/thesis/envs/diffusion_policy` voordat je `fm-smoke` indient.
