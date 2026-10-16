# Zes 2D-modellen op de robot

Voor de korte **ROBOT / EIGEN**-opstartvolgorde en attempts opnemen:
[KAS_START.md](KAS_START.md).

Alle ontwikkeling gebeurt lokaal; code gaat via commit/push en daarna pull naar
DelftBlue. Onderstaande downloads lezen alleen trainingsresultaten van DelftBlue.
Download bij voorkeur nadat de geselecteerde trainingen zijn gestopt. Downloaden
tijdens training kan ook, maar een checkpoint dat tijdens de overdracht verandert
wordt geweigerd; probeer die variant dan opnieuw.

## 1. Checkpoints ophalen op de laptop

Vanuit `~/scara_ws/greenhouse-scara`:

```bash
conda activate act
cd ~/scara_ws/greenhouse-scara
python scripts/fetch_2d_checkpoints.py
```

Dit haalt alleen `act_rgb`, `act_rgbd`, `dp_rgb`, `dp_rgbd`, `fm_rgb` en `fm_rgbd`
op. Het script opent één gedeelde SSH-verbinding: één aanmelding voor alle zes.
Het toont bij iedere variant de gekozen run.

Het script kiest per variant de hoogste Slurm-job-ID en vereist een volledig
opgeslagen checkpoint uit die run. Bij ACT moet `policy_last.ckpt` bestaan samen met `config.pkl` en
`dataset_stats.pkl`. Bij DP/Flow moet `checkpoints/latest.ckpt` bestaan. Deze
bestanden worden in de huidige training pas na de eerste trainingsepoch gemaakt.
Ontbrekende bestanden worden gemeld; het script valt niet stilzwijgend terug op
een oudere job. De exitcode is 1 als niet alle gevraagde varianten beschikbaar zijn.

### Na het stoppen van de zes trainingen

In de **DelftBlue-terminal** bekijk je de job-ID's:

```bash
module load slurm
squeue -u "$USER" -o "%.18i %.12T %.40j"
```

Stop alleen de zes bedoelde trainingen met `scancel JOBID1 JOBID2 JOBID3 JOBID4 JOBID5 JOBID6`
(vervang deze placeholders door hun numerieke job-ID's). Controleer met
`squeue -u "$USER"` dat deze jobs verdwenen zijn voordat je downloadt.
Een terminal sluiten stopt Slurm-jobs niet.

Gebruik op de **laptop** eventueel dezelfde zes ID's om de bron expliciet te kiezen:

```bash
python scripts/fetch_2d_checkpoints.py --job-ids JOBID1 JOBID2 JOBID3 JOBID4 JOBID5 JOBID6
```

Zonder `--job-ids` kiest het script de nieuwste job per variant. Het haalt de laatst
volledig opgeslagen `last`/`latest` op, niet noodzakelijk de laatste epoch in het
log. De trainers schrijven eerst een `.tmp` en vervangen daarna het checkpoint;
een onderbroken save laat het vorige complete checkpoint staan. `.tmp`-bestanden
worden niet gedownload. ACT bewaart `last` standaard iedere 10 epochs en bij een
normale afronding; DP/Flow iedere epoch.

Je kunt afzonderlijke varianten ophalen:

```bash
python scripts/fetch_2d_checkpoints.py --variants act_rgb dp_rgb
```

SSH/rsync gebruiken jouw eigen aanmelding. Als SSH-authenticatie vanuit deze
terminal niet werkt maar de SFTP-map in de bestandsbeheerder al verbonden is:

```bash
python scripts/fetch_2d_checkpoints.py --source-root "/run/user/1000/gvfs/sftp:host=login.delftblue.tudelft.nl,user=tomoyamartynow/scratch/tomoyamartynow/thesis/runs/2d"
```

Iedere download krijgt een nieuwe map onder `checkpoints/2d/<variant>/`.
`current` verwijst pas naar die map wanneer alle benodigde bestanden ontvangen
en gecontroleerd zijn. Het script vergelijkt bestandsgrootte en SHA256 met de bron,
controleert de checkpointcontainer (inclusief CRC) en pickle-bestanden, en controleert
dat de bron tijdens de overdracht niet veranderde. Bij een fout blijft de bestaande
`current` voor die variant staan. Eerdere downloads blijven staan. `download_source.json` vermeldt de bron,
jobnaam, tijd en SHA256-hashes. Er wordt niets op DelftBlue gewijzigd.
ACT downloadt `last`: `best` kan na slechts één epoch nog ongetraind zijn.

## 2. Offline controleren met dezelfde code als live

Activeer de bestaande Python 3.10-omgeving met Torch, torchvision, Hydra,
diffusers, h5py, OpenCV en dill (lokaal getest in `conda activate act`).

```bash
conda activate act
cd ~/scara_ws/greenhouse-scara
python scripts/test_2d_checkpoints.py
```

Dit test alle zes varianten achtereenvolgens op
`../datasets/green_house_dataset1_from11`, zonder robot of camera's te verbinden.
De werkelijke live-loader en voorspelfunctie worden gebruikt, met batch 1.
Er wordt gekeken of checkpoint/modality/vijf dimensies kloppen en of acties
finite zijn. JSON-uitvoer staat in `runs/checkpoint_tests/<variant>.json` met
inference-tijden, voorbeeldacties en MSE per dimensie in m²/rad².
Dit is een functionele proef op opnames, geen onafhankelijke kwaliteitsmeting.

```bash
python scripts/test_2d_checkpoints.py --variants dp_rgb fm_rgb --samples 4
# Alleen als CUDA op deze laptop beschikbaar is:
python scripts/test_2d_checkpoints.py --device cuda:0
```

Voor een snelheidsmeting met twee opwarmvoorspellingen en tien gemeten
voorspellingen per model, met dezelfde CPU-threadinstelling als de live-runner:

```bash
OMP_NUM_THREADS=1 python scripts/test_2d_checkpoints.py --device auto \
  --warmup 2 --samples 10 --output runs/inference_benchmark
```

De JSON-bestanden bevatten gemiddelde, mediaan en p95 in milliseconden, het
gebruikte device en het aantal inferentiestappen. De timing omvat beeldverwerking,
model-inference en het terughalen van acties naar de CPU. Checkpoint laden,
dataset lezen, camera's en robotcommunicatie vallen buiten deze meting.

Voor een uitgebreidere snelheidsvergelijking op de GPU:

```bash
python scripts/benchmark_2d_speed.py --device cuda:0
```

Dit probeert 1, 2 en 4 CPU-threads, meet beeldencoder en sampling apart voor
DP/Flow, en vergelijkt verschillende aantallen inferentiestappen. Flow wordt met
Heun én Euler getest: Heun doet twee U-Net-berekeningen per stap, Euler één.
De resultaten staan in `runs/benchmark_2d_speed`. De proef verandert geen
checkpoint of live-instellingen. Minder stappen en een andere solver veranderen
wel de voorspellingen; gemeten snelheid bewijst geen succesvolle robotuitvoering.

Een vroege checkpoint kan technisch werken en toch slechte acties voorspellen.
De default DP/Flow-inferentiestappen worden uit het checkpoint gelezen; met
`--num-inference-steps` kun je de snelheid/nauwkeurigheid afwegen.

## 3. Hardware-instellingen

Alle zes gebruiken `config/robot_2d.yaml`. De defaults komen uit
`src/scara_dataset/record_dataset.py`:

- robot `192.168.1.10:10100`;
- `camera_top`: `130322273728`;
- `camera_bottom`: `128422271208`;
- beide camera's op 640×480 bij 30 Hz; de modelspecifieke preprocessing komt uit
  het checkpoint (RGBD inclusief uitlijning, depth scale en geldigheidsmasker);
- schaarfeedback `/scissor/rotation` en commando `/scissor/set_rotation`.

De camera-serienummers zijn aanpasbaar in hetzelfde bestand. Op jouw verzoek
staat `check_joint_limits: false`: de oude vaste J1–J4-positiegrenzen worden niet
afdwongen. Ze blijven in de configuratie voor wanneer je ze opnieuw opmeet.
De schaargrenzen, finite-controle en `max_delta` per commando blijven actief.
De oude J1–J4-grenzen botsen namelijk met delen van de opnames (J1 tot 1,138 m,
J2 tot −1,641 rad, J3 tot 0,145 rad). Er wordt niet automatisch geclipt of naar
een beginpose bewogen. Zet de robot nabij de beginpose van de demonstratie.

De vijf waarden zijn **J1 (meter), J2/J3/J4 (radiaal), scissor_roll (radiaal)**.
De vijfde observatie is de gemeten schaarhoek. De vijfde actie is een doelhoek,
zoals opgeslagen in `actions/scissor_rotation`; deze wordt niet naar graden
omgerekend voordat hij naar ROS gaat. J1–J4 worden via de bestaande TCS-omrekening
naar millimeters/graden verstuurd. De schaaractie roteert de schaar volgens de
bestaande driver; een eventuele aparte knip-/sluitactuator wordt hiermee niet
extra aangestuurd.

Op de schaarcomputer moet de bestaande ROS2-node draaien. Op de laptop:

```bash
conda activate act
source /opt/ros/humble/setup.bash
source ~/scara_ws/install/setup.bash
# Gebruik dezelfde ROS_DOMAIN_ID als op de schaarcomputer.
ros2 topic echo /scissor/rotation --once
ros2 service list
```

Beide machines hebben de gebouwde `scissor_interfaces/srv/SetRotation` nodig.
Je hebt bevestigd dat op de schaarcomputer de recorder-services draaien,
inclusief `/scissor/halt`. Daarom staat `stop_mode: halt` ingesteld en vereist de
runner de halt-service vóór uitvoering. De lokale voorbeeldkopie van
`scissor_pc/scissor_node.py` mist die service; gebruik de werkende node op de
robotcomputer. De optionele standen `auto`/`hold` ondersteunen een houdcommando
naar de verse gemeten positie; dat is geen hardware-noodstop.

## 4. Live lezen en voorspellen

Start één variant tegelijk. Bijvoorbeeld:

```bash
bash scripts/robot/act_rgb.sh --chunks 10 --show-camera
```

Dit leest beide camera's, J1–J4 en de werkelijke schaarstand. Het toont de
voorspellingen en grenscontroles, zonder `MoveJ` of schaarcommando's te versturen.
Voor de andere modellen vervang je de scriptnaam:

| Model | RGB | RGBD |
| --- | --- | --- |
| ACT | `scripts/robot/act_rgb.sh` | `scripts/robot/act_rgbd.sh` |
| DP | `scripts/robot/dp_rgb.sh` | `scripts/robot/dp_rgbd.sh` |
| Flow | `scripts/robot/fm_rgb.sh` | `scripts/robot/fm_rgbd.sh` |

Live kiest standaard `--device auto`: CUDA wanneer beschikbaar, anders CPU.
Met `--device cuda:0` of `--device cpu` kun je dit expliciet kiezen. Een andere checkpoint kan met `--checkpoint /volledig/pad/model.ckpt`;
ACT vereist `config.pkl` en `dataset_stats.pkl` in dezelfde map.

## 5. Daadwerkelijk uitvoeren

Dezelfde scripts voeren bewegingen uit wanneer je `--execute` toevoegt:

```bash
bash scripts/robot/act_rgb.sh --execute --chunks 10 --show-camera
bash scripts/robot/act_rgbd.sh --execute --chunks 10 --show-camera
bash scripts/robot/dp_rgb.sh --execute --chunks 10 --show-camera
bash scripts/robot/dp_rgbd.sh --execute --chunks 10 --show-camera
bash scripts/robot/fm_rgb.sh --execute --chunks 10 --show-camera
bash scripts/robot/fm_rgbd.sh --execute --chunks 10 --show-camera
```

Kies één van deze commando's. ACT (RGB en RGBD) voert standaard de eerste 15 acties
van de chunk van 30 uit, neemt daarna nieuwe observaties en voorspelt opnieuw.
DP/Flow gebruiken standaard één actie per voorspelling en geven 8 uitvoerbare
acties terug. Met `--action-count N` kun je dit aanpassen.
`--hz 30` bepaalt de pauze tussen
acties binnen een chunk; cameralezen, model-inference en netwerkverkeer kosten
ook tijd, dus dit garandeert geen totale regelkring van 30 Hz.

Ctrl+C of `q` in het cameravenster stopt de loop. Met `s`/`f` markeer je succes/mislukking. Na uitvoering probeert de runner
zowel de robot als de schaar te stoppen. Ontbrekende/verouderde schaarfeedback,
te oude camerabeelden, niet-finite acties en overschreden ingeschakelde positie-/stapgrenzen
breken uitvoering af. Bij een weggevallen netwerk kan software stoppen niet
garanderen; de fysieke stop blijft nodig.

Zonder attempt-opname staat per sessie een JSONL-log in `runs/robot/`, inclusief configuratie,
checkpointpad, gemeten vijfdimensionale toestand, voorspellingen, verzonden
acties en inference-tijden. De bestaande `scripts/run/*.sh` blijven de oude
**offline** evaluaties; `scripts/robot/*.sh` zijn de nieuwe live-runners.

## Attempts opnemen

```bash
bash scripts/attempt.sh act_rgb poging_01 --execute
```

Vervang `act_rgb` door de gewenste variant. Zonder `--execute` neem je een
leesmodus-poging op. Elke poging krijgt een eigen map in `runs/attempts/` met
beide RGB-video's, gemeten standen, voorspelde en aangevraagde/afgeronde acties,
checkpointgegevens en resultaat. Met `--record-depth` bewaar je bij RGBD ook de
ruwe diepte. De camera-opname loopt op de achtergrond door tijdens inference.
Zie [KAS_START.md](KAS_START.md) voor de bediening en bestandsindeling.

## Lokale verificatie

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Deze tests gebruiken tijdelijke checkpoints en nagebootste robot-/schaarclients.
Ze sturen geen hardware aan. De ronde door echte ACT/DP/Flow-modelklassen is voor
alle zes varianten getest met kleine modellen op CPU. De gedownloade volledige
checkpoints moeten met stap 2 worden gecontroleerd; hardwaregedrag moet bij de
robot worden gecontroleerd.
