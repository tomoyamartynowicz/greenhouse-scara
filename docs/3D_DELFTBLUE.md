# ACT3D, DP3 en Flow3D — één camera, XYZRGB en schaar

Alle codewijzigingen gebeuren lokaal. Commit/push zelf, daarna `git pull` op
DelftBlue. Er zijn door deze implementatie geen clusterjobs gestart.

## Wat wordt getraind?

Alle drie gebruiken dezelfde **camera_top**, **1.024 gekleurde XYZRGB-punten** en
vijf observatie-/actiewaarden: **J1, J2, J3, J4, scissor_roll**. J1 is in meters,
de overige waarden in radialen. De schaarobservatie is gemeten; de schaaractie
komt uit `actions/scissor_rotation`. Jointtargets volgen de bestaande
`action_source: auto` / volgende-qpos-conventie. Episodes vanaf 11 worden gebruikt;
80% trainen, 20% valideren, seed 0, dezelfde episodesplit voor de drie modellen.

| Model | Architectuur | Voorspelhorizon | Observaties | Uitvoeren per voorspelling |
|---|---|---:|---:|---:|
| ACT3D | ACT CVAE/transformer + gepoolde DP3 PointNet-feature | 30 | 1 | 15 |
| DP3 | DP3 PointNet + conditionele U-Net, DDIM 10 stappen | 16 | 2 | 8 |
| Flow3D | Dezelfde PointNet/U-Net, flow matching, Heun 8 stappen | 16 | 2 | 8 |

DP3 hergebruikt de al aanwezige vendored implementatie van
[3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy), inclusief
de U-Net met kanalen `[512, 1024, 2048]`. Flow3D leert een velocity field en
integreert dit met Euler/Heun. ACT3D gebruikt de bestaande ACT-transformer en
CVAE met een PointNet-encoder. De 2D-modelarchitecturen zijn niet aangepast.

Defaults: batch 64, vier dataloaderworkers, maximaal **10 uur per trainingsjob**,
100.000 epochs als bovengrens, constante learning rate na 500 warmupstappen.
ACT3D: lr 5e-5, KL 10. DP3/Flow3D: lr 1e-4, EMA.
Iedere voltooide epoch schrijft atomisch `latest.ckpt`; `best.ckpt` volgt de beste
validatieloss. Iedere tien epochs wordt een backup gemaakt, waarvan de laatste
vijf blijven staan. `latest.ckpt` bevat ook optimizer-/scheduler-/RNG-state voor
resume. Een tijdslimiet kan een lopende epoch onderbreken; de laatste voltooide
checkpoint blijft beschikbaar. Deze volledige checkpoints kunnen groot zijn.

## Kalibratie en pointclouds

Configuratie: `calibration/greenhouse_top_approximate.yaml`.

- Op jouw verzoek gebruiken we intrinsics van **D405 130322273198** als
  **benadering** voor opnamecamera_top **130322273728**. Dit is geen gemeten
  kalibratie van camera_top; metrische posities kunnen daardoor afwijken.
- De HDF5-depth is al op RGB uitgelijnd. We gebruiken daarom de Color-intrinsics
  en de bijbehorende distortioncoëfficiënten, met de depth scale uit de HDF5.
  Kleur wordt van dezelfde pixel gelezen. De interne depth→color-transformatie
  wordt niet nogmaals toegepast.
- Coördinaten zijn het optical frame van camera_top: X rechts, Y omlaag,
  Z vooruit. Geen robotextrinsics of tweede-camera-transformatie nodig.
- We behouden depth van **0,02–1,0 meter**, nemen iedere tweede pixel en selecteren
  1.024 punten met deterministische farthest-point sampling uit maximaal 4.096
  gelijkmatig gekozen kandidaten. XYZ blijft in meters; RGB begint in [0,1].
  Per-kanaalnormalisatie wordt alleen op de trainingsepisodes gefit.
- Deze afstandsselectie verwijdert een groot deel van de kasachtergrond.
  Pas `pointcloud` in **alle drie** de trainingsconfiguraties samen aan als het
  werkgebied groter moet zijn; bouw dan een nieuwe cachemap.
- Het checkpoint bewaart de volledige kalibratie, cropping, sampling en
  normalisatie. Live gebruikt exact die instellingen. Bij deze expliciet
  geaccepteerde benadering meldt de runner afwijkende echte intrinsics en blijft
  de trainingsprojectie gebruiken; bij normale gemeten kalibraties is een
  verschil een fout.

De nieuwe trainingsprofielen zijn:

```text
src/greenhouse_scara_3d_act/config.yaml
src/greenhouse_scara_3d_diffusion_policy/config_greenhouse.yaml
src/greenhouse_scara_3d_flow_matching_policy/config_greenhouse.yaml
```

De oudere DP3/Flow-configs blijven beschikbaar voor eerdere vier-jointproeven.

## Eenmalig: omgeving controleren

De jobs gebruiken dezelfde bestaande Pixi-omgevingen als 2D. Voor offline
deprojectie is de RealSense-Pythonbibliotheek nodig, zonder aangesloten camera.
Controleer op DelftBlue:

```bash
pixi run --manifest-path /scratch/$USER/thesis/envs/act/pixi.toml \
  python -c 'import torch, torchvision, diffusers, einops, omegaconf, h5py, termcolor, pyrealsense2'
pixi run --manifest-path /scratch/$USER/thesis/envs/diffusion_policy/pixi.toml \
  python -c 'import torch, torchvision, diffusers, einops, omegaconf, h5py, termcolor, pyrealsense2'
```

Als `termcolor` of `pyrealsense2` ontbreekt, voeg die toe **nadat de huidige jobs
die deze omgeving gebruiken klaar zijn**:

```bash
pixi add --manifest-path /scratch/$USER/thesis/envs/act/pixi.toml --pypi termcolor 'pyrealsense2==2.58.2.10647'
pixi add --manifest-path /scratch/$USER/thesis/envs/diffusion_policy/pixi.toml --pypi termcolor 'pyrealsense2==2.58.2.10647'
```

## Starten op DelftBlue

In de **remote terminal**, vanuit de repository:

```bash
module load slurm
cd ~/scara_ws/greenhouse-scara
git pull

# Eén voorbereiding voor alle drie modellen; bewaart originele HDF5's ongewijzigd.
sbatch scripts/slurm/prepare_3d.slurm
```

Dit leest `/scratch/$USER/thesis/datasets/green_house_dataset1` en maakt compacte
pointcloud-bestanden in
`/scratch/$USER/thesis/datasets/green_house_dataset1_pointcloud_top`.
Vier CPU-processen, geen GPU. Voltooide episodes worden bij opnieuw starten
hergebruikt. Wijzigingen aan bronbestanden/kalibratie/preprocessing worden
gedetecteerd, zodat een verkeerde cache niet stilzwijgend wordt gebruikt.

**Als voorbereiding succesvol klaar is:**

```bash
sbatch scripts/slurm/act3d.slurm
sbatch scripts/slurm/dp3.slurm
sbatch scripts/slurm/fm3.slurm
```

Of zet ze meteen in de wachtrij achter de voorbereiding. Vervang `123456`
door het jobnummer van `prepare_3d`:

```bash
sbatch --dependency=afterok:123456 scripts/slurm/act3d.slurm
sbatch --dependency=afterok:123456 scripts/slurm/dp3.slurm
sbatch --dependency=afterok:123456 scripts/slurm/fm3.slurm
```

Log: `~/scara_ws/greenhouse-scara/slurm-JOBNUMMER.out`.
Runs: `/scratch/$USER/thesis/runs/3d/MODEL_xyzrgb_top_scissor_JOBNUMMER/`.
Daar staan ook `config.yaml` en `metrics.jsonl`.

Voor een korte clustercontrole met de volledige architectuur:

```bash
GREENHOUSE_3D_SMOKE=1 sbatch --time=00:30:00 scripts/slurm/act3d.slurm
GREENHOUSE_3D_SMOKE=1 sbatch --time=00:30:00 scripts/slurm/dp3.slurm
GREENHOUSE_3D_SMOKE=1 sbatch --time=00:30:00 scripts/slurm/fm3.slurm
```

Dit gebruikt één trainbatch en één validatiebatch, batchsize 1, zonder EMA.
Voor een andere echte batchsize: bijvoorbeeld
`GREENHOUSE_3D_BATCH_SIZE=32 sbatch scripts/slurm/dp3.slurm`.

## Ophalen en offline testen op je laptop

Na training, met echte jobnummers in de paden:

```bash
conda activate act
cd ~/scara_ws/greenhouse-scara
python scripts/fetch_checkpoints.py \
  --act3d /scratch/tomoyamartynow/thesis/runs/3d/act3d_xyzrgb_top_scissor_JOBNUMMER/latest.ckpt \
  --dp3 /scratch/tomoyamartynow/thesis/runs/3d/dp3_xyzrgb_top_scissor_JOBNUMMER/latest.ckpt \
  --fm3 /scratch/tomoyamartynow/thesis/runs/3d/fm3_xyzrgb_top_scissor_JOBNUMMER/latest.ckpt

python scripts/test_3d_checkpoints.py --device cuda:0
```

De lokale bestanden komen in `checkpoints/act3d`, `checkpoints/dp3` en
`checkpoints/fm3`. Deze checkpoints bevatten de kalibratie en normalizer, dus
geen losse trainingsbestanden nodig. De test gebruikt de originele lokale
demonstraties en de live pointcloudberekening, zonder ROS, camera of robot.
JSON-resultaten: `runs/checkpoint_tests_3d`.

## Robot en attempts

Gebruik dezelfde **ROBOT / EIGEN** ROS-opstartcommando's uit
[KAS_START.md](KAS_START.md). De 3D-runner opent alleen camera_top; de serial en
resolutie komen uit het checkpoint. Begin met meekijken:

```bash
bash scripts/attempt.sh act3d controle_01 --device cuda:0
bash scripts/attempt.sh dp3 controle_01 --device cuda:0
bash scripts/attempt.sh fm3 controle_01 --device cuda:0
```

Kies één model. Voeg `--execute` toe om robot en schaar aan te sturen. Dezelfde
toetsen werken: `s` geslaagd, `f` mislukt, `q` stoppen. Video, robot-/schaarstate,
voorspellingen en verzonden acties komen in `runs/attempts/`. Met `--record-depth`
bewaar je ook de depthbeelden. Voer de datasetrecorder niet gelijktijdig uit.
`--action-count` overschrijft het aantal uit te voeren acties; `--num-inference-steps`
past bij DP3/Flow3D het samplingbudget aan.

## Lokaal gecontroleerd

- Geometrie, XYZRGB, vijf dimensies, scissor-targetuitlijning, cachecontrole,
  gradienten, checkpointreload en overeenstemming tussen dataset- en live-input.
- Korte CPU-training voor alle drie modellen op episodes 11 en 12, met kleinere
  netwerken en één train-/validatiebatch om de volledige gegevensketen te testen.
- De smoke-checkpoints zijn daarna via de offline CLI op echte beelden getest.
- Ook de volledige architecturen zijn met echte pointcloud-input op de Quadro
  P1000 gecontroleerd: geldige vijf-dimensionale voorspellingen. Dit was inference
  met willekeurig geïnitialiseerde gewichten, geen volledige training.
- Dit is geen getrainde policy voor deployment en geen meting van robotsucces.
  Volledige training, het clustergeheugengebruik en echte robotuitvoering moeten
  nog op DelftBlue/in de kas worden beoordeeld.
