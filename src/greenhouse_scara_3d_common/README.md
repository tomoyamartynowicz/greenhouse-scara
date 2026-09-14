# Gedeelde 3D-invoer

Deze map en de DP3-adapter zijn nu onderdeel van `greenhouse-scara`. Begin bij de
[actuele DP3-handleiding](../greenhouse_scara_3d_diffusion_policy/README.md) en
[inspect_pointcloud.ipynb](../../inspect_pointcloud.ipynb). De aangeleverde
D405-dump is geïmporteerd naar `calibration/camera_top_d405_aligned.yaml`.
Voor DP3 staan crop/afstandsgrens/kandidaatlimiet nu uit en is limits-normalisatie
actief. De overige 3D-adapters blijven vooralsnog buiten deze repository in de
lokale `scara_ws/src`; onderstaande tekst beschrijft ook die eerdere adapters.

# SCARA: vier puntenwolkmodellen

De bestaande ACT-, diffusion-, flow- en recorderprojecten blijven ongewijzigd. Deze vier nieuwe projecten delen alleen de HDF5-loader, 3D-voorbewerking en trainingslus:

| Map in `src/` | Model |
| --- | --- |
| `greenhouse_scara_3d_diffusion_policy` | Officiële DP3 PointNet + diffusion U-Net |
| `greenhouse_scara_3d_flow_matching_policy` | Huidige SCARA flow-matching U-Net en loss/sampler + DP3-encoder |
| `greenhouse_scara_3d_act` | Eigen ACT-variant: DP3 PointNet-feature + ACT CVAE/transformer |
| `greenhouse_scara_maniflow` | Officiële ManiFlow: pointwise DP3-encoder + DiTX + flow/consistency-loss |

Dit zijn SCARA-adaptaties, geen volledige reproducties van de papers. De 3D-flowvariant volgt het idee van de **3D Flow Matching Policy*** baseline in [ManiFlow §3.1](https://arxiv.org/html/2509.01819v1#S3.SS1): een DP3-encoder toevoegen aan image-based flow matching. Hij gebruikt de huidige SCARA-flowimplementatie, niet de aparte FlowPolicy consistency-methode. ManiFlow zelf is een ander model en trainingsdoel.

## Eenmalig voorbereiden

Vanuit `/home/tomoya/scara_ws`, met de bestaande `act`-omgeving:

```bash
conda activate act
python -m pip install -r src/greenhouse_scara_3d_common/requirements.txt
python src/greenhouse_scara_3d_common/export_calibration.py
```

Het kalibratiescript opent alleen `camera_top` (serial `130322273198`), RGB + depth op 640×480/30 Hz. Sluit andere camera-applicaties eerst. Het leest de intrinsics van het **verwerkte depthframe na RGB-alignment** en vult `calibration.yaml`. Bestaande gemeten kalibratie wordt niet overschreven; gebruik dan `--output ander_pad.yaml`.

Het voorbeeld `datasets/test_rgbd_single_joints/episode_0.hdf5` bevat ruwe depth, de schaal naar meters en de alignment-vlag, maar **geen intrinsics**. Daarom bevat het meegeleverde kalibratiebestand lege waarden en stopt training zolang die ontbreken. Lees de kalibratie uit met dezelfde fysieke camera, resolutie en alignment als de opname. Het HDF5-bestand bevat geen serial om deze identiteit achteraf automatisch te controleren. Zonder de juiste kalibratie zijn bestaande depthbeelden niet betrouwbaar naar metrische punten om te zetten.

Aligned depth is bruikbaar: deprojectie gebruikt de intrinsics van die aligned pixelgrid, niet blind de native depth-intrinsics. Gaten en ongeldige depth worden weggefilterd. Alignment kan oorspronkelijke samples verliezen; dit reconstrueert niet exact de oorspronkelijke native puntenwolk. Zie de [RealSense-projectiedocumentatie](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0).

## Trainen

Voor een technische test zonder gemeten depth is er ook de [dummy-dataset met twintig demonstraties](../../datasets/leaf_cutting_dummy_rgbd_20/README.md). Deze combineert echte RGB/joints met synthetische depth en bevat aparte volledige en kleine CPU-testconfiguraties voor alle vier de modellen. Gebruik de bijbehorende synthetische kalibratie alleen voor die dataset. `make_dummy_dataset.py` maakt deze dataset uit `datasets/leaf_cutting_experiment_rgb` in een nieuwe map en weigert bestaande output te overschrijven.

Pas `dataset_dir` en eventueel `output_dir` in de `config.yaml` van het gekozen model aan. Daarna:

```bash
python src/greenhouse_scara_3d_diffusion_policy/train.py
python src/greenhouse_scara_3d_flow_matching_policy/train.py
python src/greenhouse_scara_3d_act/train.py
python src/greenhouse_scara_maniflow/train.py
```

Kies één commando per training. Elk project heeft eigen instellingen en een eigen outputmap. Relatieve dataset-, output- en kalibratiepaden zijn relatief aan `scara_ws`. Optionele overrides:

```bash
python src/greenhouse_scara_3d_diffusion_policy/train.py dataset_dir=datasets/mijn_demos output_dir=runs/dp3/test01 batch_size=4
```

Met alleen het voorbeeldbestand kun je een technische proef doen, nadat kalibratie is ingevuld:

```bash
python src/greenhouse_scara_3d_diffusion_policy/train.py allow_single_episode=true epochs=1 max_train_steps=1 max_val_steps=1 output_dir=runs/dp3/smoke
```

Dan worden dezelfde demonstratiebeelden voor training én validatie gebruikt; de score zegt niets over generalisatie. Normale training vereist minstens twee episodes en splitst per episode. ManiFlow vereist train- en validatiebatches deelbaar door vier en minimaal vier samples, vanwege de 75/25 flow/consistency-splitsing. Grote standaardmodellen kunnen veel GPU-geheugen vragen; de automatische tests gebruiken kleinere modellen op CPU.

Output: `config.yaml`, `metrics.jsonl`, `latest.ckpt` en `best.ckpt`. `best` wordt gekozen op validatieloss. Checkpoints bewaren ook kalibratie, normalisatie, datasplit, optimizer, scheduler, EMA en random states. Een bestaande run hervatten:

```bash
python src/greenhouse_scara_3d_diffusion_policy/train.py output_dir=runs/dp3/test01 resume=true epochs=300
python src/greenhouse_scara_3d_diffusion_policy/eval.py runs/dp3/test01/best.ckpt --device cuda:0 --max-batches 10
```

Gebruik bij hervatten dezelfde dataset- en modelinstellingen als de oorspronkelijke run. `epochs` is het totale aantal epochs. `eval.py` evalueert offline de opgeslagen validatiesplit; het bestuurt geen robot. MSE wordt per joint in de oorspronkelijke eenheden gerapporteerd. Losses van verschillende modeltypen zijn niet rechtstreeks vergelijkbaar. De gemengde `sampled_action_mse` in de trainingslog is alleen een technische indicatie.

## Dataset en acties

De loader leest `episode_*.hdf5` rechtstreeks, zonder bestanden te veranderen:

```text
observations/qpos                 [T, 4]
observations/depth/camera_top      [T, H, W] uint16
  attrs: depth_scale, aligned_to_rgb
observations/images/camera_top     [T, H, W, 3] uint8 (nodig bij use_color=true)
action                            [T, 4] (optioneel)
```

Elke geconfigureerde camera heeft dezelfde groepen. Frames en joints worden per opgeslagen rij gekoppeld; deze loader voert geen nieuwe tijdsynchronisatie uit. Controleer de opname met de bestaande datasetnotebook. Trainen vereist joints; camera-only testdatasets zijn hiervoor niet geschikt.

Met `action_source: auto` gebruikt de loader bestaande `action`-data of leidt hij acties af uit joints: `action[t] = qpos[min(t + action_offset, T - 1)]`. Standaard is de offset één frame. Gebruik `action_source: next_qpos` om altijd joints te gebruiken. De verschuiving gebeurt **vóór** het maken van chunks en hangt dus niet af van de chunklengte. Dit zijn toekomstige gemeten posities, niet noodzakelijk de oorspronkelijke robotcommando's.

`horizon` bepaalt de trainingschunk; `n_obs_steps` de observatiegeschiedenis en `n_action_steps` de lengte van de uitgegeven actiechunk. De chunk begint bij de eerste observatie; de uitgegeven acties beginnen bij de laatste observatie. Aan episodegrenzen wordt de randwaarde herhaald; ACT krijgt daarnaast een paddingmasker. ACT gebruikt één observatie zoals de bestaande variant. Joint- en actienormalisatie wordt uitsluitend op trainingsepisodes gefit; XYZ blijft in meters.

## Eén of twee camera's, optioneel kleur

```yaml
pointcloud:
  cameras: [camera_top]  # of [camera_top, camera_bottom]
  reference_frame: camera_top
  num_points: 1024
  use_color: false      # true: XYZRGB in plaats van XYZ
  min_depth_m: 0.02
  max_depth_m: 1.5
  pixel_stride: 2
  fps_candidates: 4096
  crop_min: null        # eventueel [xmin, ymin, zmin], in meters
  crop_max: null
```

Depth wordt op zijn gekalibreerde resolutie gedeprojecteerd, niet eerst naar een kleiner beeld geschaald. `pixel_stride` slaat pixels over met behoud van hun oorspronkelijke pixelcoördinaten. Vervolgens: ongeldige/verre punten verwijderen → cameratransformaties → optionele XYZ-crop → vast aantal punten. Deterministische farthest-point sampling gebruikt maximaal `fps_candidates` kandidaten; dit is een praktische aanpassing ten opzichte van FPS op alle punten. Bij te weinig geldige punten worden punten herhaald; bij nul geldige punten volgt een fout. Een kleine geheugen-cache voorkomt een deel van de herberekeningen; er worden geen grote puntenwolkbestanden opgeslagen.

Met `use_color: true` krijgt ieder punt RGB-waarden in [0, 1] uit het corresponderende aligned RGB-pixel. Dit is **geen aparte RGB-image-encoder**. Er is geen willekeurige beeld- of puntenwolkaugmentatie toegevoegd. Alleen XYZ kan ook met native depth: exporteer dan met `--raw-depth` en gebruik een dataset die zonder alignment is opgenomen.

Voor twee camera's:

```bash
python src/greenhouse_scara_3d_common/export_calibration.py --camera camera_top 130322273198 --camera camera_bottom 123622270497 --output src/greenhouse_scara_3d_common/calibration_dual.yaml
```

Stel vervolgens `calibration_file` in en voeg de **gemeten extrinsieke kalibratie** toe aan `camera_bottom.to_reference`: een rigide 4×4 matrix die bottom-camera-XYZ naar top-camera-XYZ omzet, met translatie in meters. Voor de referentiecamera betekent `null` de identiteitsmatrix. Het exportscript meet deze onderlinge pose niet. Twee camera's mogen niet zonder die transformatie worden samengevoegd. Bij verandering van cameraposities is nieuwe extrinsieke kalibratie nodig. Eén camera in zijn eigen frame vereist deze stap niet.

## Live input en validatie

`live.py` bevat `LivePointCloudSource`: dezelfde voorbewerking voor live RealSense-beelden en aangeleverde joints. `runtime.load_checkpoint()` laadt een policy met de opgeslagen kalibratie en normalisatie. De adapter controleert het live camerprofiel; de aanroeper moet de timing van joints en beelden verzorgen. Er is nog geen complete robotuitvoerlus met actie-uitvoering voor deze vier nieuwe modellen.

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s src/greenhouse_scara_3d_common/tests -v
```

De tests gebruiken synthetische gekalibreerde HDF5-bestanden en kleine CPU-modellen: geometrie, twee-camera-transformaties, acties/chunks, XYZ/XYZRGB, gradients, sampling, checkpoints en hervatten. Ze bewijzen geen kwaliteit op echte demonstraties en testen geen camerhardware. Zie [vendor/README.md](vendor/README.md) voor de herkomst van de modelcode.
