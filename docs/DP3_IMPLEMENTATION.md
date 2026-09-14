# DP3: wat volgt upstream en wat is lokaal?

Referentie: [officiële DP3-config op de vastgelegde commit](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml).

De PointNet, conditionele U-Net en diffusion-policy/loss/sampler zijn overgenomen.
De bestandsvergelijking van deze drie kernbestanden staat in `dp3_source_audit.json`.
Dit is de volledige DP3-architectuur, niet de aparte `simple_dp3`-variant.
Encoder-output 64, U-Net `[512,1024,2048]`, horizon 16, twee observaties, acht acties,
DDIM met tien inference-stappen/100 trainingstappen, `prediction_type: sample`,
FiLM-conditionering en AdamW-betas `[0.95,0.999]` volgen de officiële configuratie.

Lokale aanpassingen:

- Vier SCARA-joints en absolute targets; ontbrekende acties worden afgeleid als
  volgende gemeten qpos. De HDF5 blijft alleen-lezen.
- Eigen HDF5-loader en trainingslus, geen simulators/Zarr-conversie nodig.
  Padding en windowselectie zijn lokaal; de trainer is geen exacte kopie van upstream.
- Geen werkgebiedcrop of afstandscrop. Die is data-/taakspecifiek en geen vereiste
  voor de PointNet/U-Net. Validiteit (nuldepth verwijderen) blijft nodig.
- Deterministische NumPy-FPS op alle punten, in plaats van de upstream GPU-FPS.
  Het beginpunt wordt deterministisch gekozen; exacte indices kunnen verschillen.
  Een kandidaatlimiet is optioneel en staat voor DP3 uit.
- Per-as limits-normalisatie zoals de upstream Adroit-dataloader, met bounds uit
  alle geldige punten van alleen trainingsepisodes. Upstream fit op zijn opgeslagen
  gesamplede dataset; deze loader gebruikt streaming dense bounds en vermijdt
  validatielekkage. Beeldweergave en PLY blijven in meters.
- Kleine instelbare batch, eigen epoch-aantal, expliciete single-episode-test,
  aanvullende validatie/checkpoints. Andere 3D-modeladapters zijn geen onderdeel
  van deze DP3-training en zijn nog niet naar deze repository verhuisd.

# Calibratie en geometrie

De aangeleverde HDF5 slaat depth op als uint16, met schaal circa **0.0001 meter**,
niet 0.001. `aligned_to_rgb=True` bepaalt welke intrinsics gebruikt worden.
De RealSense-SDK geeft een aligned profiel de intrinsics van de doelstream:
[align.cpp](https://github.com/realsenseai/librealsense/blob/master/src/proc/align.cpp).
Daarom gebruiken we hier Color 640×480 met de bijbehorende inverse-Brown-Conrady-
coëfficiënten. Deprojection gebeurt met de SDK, zodat de vervorming niet wordt genegeerd.
De interne depth→color-extrinsics worden niet opnieuw toegepast op de aligned data.
Het originele tekstbestand en zijn SHA-256 staan bij de gegenereerde calibratie.

Alignment rasteriseert depth op RGB en bewaart depthwaarden; verloren samples en
occlusies kun je hieruit niet exact terughalen. Deze geometrie volgt het SDK-profiel,
geen garantie op exacte native-sensorgeometrie. Een robotpose of tweede-camera-pose
staat niet in de aangeleverde dump. Voor één camera nemen we zijn eigen optical
frame; voor twee camera's moet bottom→top apart gemeten worden.

# Controle op de echte demo

Frame 0: 307.200 pixels, 37.392 nulmetingen, 269.808 geldige punten, 1.024 modelpunten.
Er zijn 5.279 waarden gelijk aan de maximumcode 65535; deze worden gerapporteerd,
niet stilzwijgend verwijderd. De mediane positieve depth is ongeveer 2.927 m.
De kamerachtergrond maakt dus een aanzienlijk deel van de observatie uit.

`inspect_pointcloud.ipynb` en `scripts/inspect_pointcloud.py` gebruiken dezelfde
builder en FPS als training. De HTML toont een verdunde volledige cloud en alle
modelpunten; de volledige PLY bevat ieder geldig punt. De NPY bevat exact de
geometrische modelinput vóór normalisatie. Bekijk de cloud naast RGB/depth;
de viewer is bedoeld om geometrie en sampling te beoordelen, niet om automatisch
camera-identiteit of absolute meetnauwkeurigheid te bewijzen.
