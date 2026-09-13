# Greenhouse SCARA policies

Eén repository voor de drie 2D-policies en hun gedeelde HDF5/preprocessing-code.

```text
src/
  greenhouse_scara_common/
  greenhouse_scara_act/
  greenhouse_scara_diffusion_policy/
  greenhouse_scara_flow_matching_policy/
environments/requirements-train.txt
scripts/submit.sh
scripts/slurm_env.sh
```

Houd deze vier bronmappen samen. De train/eval-entrypoints voegen de gedeelde
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
ze in `/scratch/$USER/thesis/envs/act`, `diffusion_policy` of
`flow_matching_policy`. Met `GREENHOUSE_PIXI_PROJECT` kies je zelf een map met
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

**De huidige inputs blijven top-RGB + top-depth.** Voor de besproken twee-camera
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
