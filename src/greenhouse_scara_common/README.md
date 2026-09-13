# Greenhouse SCARA modellen

Drie aparte kopieën van de werkende modelcode:

- `../greenhouse_scara_act`
- `../greenhouse_scara_diffusion_policy`
- `../greenhouse_scara_flow_matching_policy`

De oorspronkelijke `scara_*`-projecten zijn niet aangepast. Alleen broncode en
configuraties zijn gekopieerd, zonder `.git`, datasets, checkpoints en caches.
De gekopieerde modelpackages hebben eigen namen. Deze map bevat de gedeelde
HDF5-invoer, beeldverwerking en camera-adapter; houd de vier mappen samen in `src`.
De normale train- en eval-entrypoints regelen het Python-importpad zelf.

## Invoer kiezen

ACT: `greenhouse_scara_act/inputs.yaml`.
Diffusion/flow: `<modelpackage>/config/task/scara_image.yaml`.
In alle drie staat dezelfde `shape_meta`-structuur. Standaard zijn de volgende
streams geselecteerd, passend bij `datasets/test_rgbd_single_joints/episode_0.hdf5`:

```yaml
shape_meta:
  obs:
    camera_top_rgb:
      camera: camera_top
      type: rgb
      shape: [3, 240, 320]
    camera_top_depth:
      camera: camera_top
      type: depth
      shape: [2, 240, 320]
      max_depth_m: 0.5
      aligned_to_rgb: true
    qpos:
      type: low_dim
      shape: [4]
  action:
    shape: [4]
```

Voeg voor de onderste camera dezelfde twee entries toe met sleutels
`camera_bottom_rgb`/`camera_bottom_depth` en `camera: camera_bottom`.
Verwijder de depth-entry voor RGB-only training. Een camera mag alleen RGB hebben
terwijl een andere ook depth heeft. Elke aangevraagde stream moet in alle demo's
staan: ontbrekende beelden worden niet stilzwijgend vervangen door nullen.

De streamnaam (`camera_top_depth`) is een modelsleutel; `camera` bepaalt de naam
in HDF5. RGB wordt gelezen uit `observations/images/<camera>`, depth uit
`observations/depth/<camera>`. Joints zijn verplicht voor policytraining.

RGB wordt naar de geconfigureerde resolutie verkleind en naar `[0,1]` geschaald;
RGB-normalisatie gebeurt vervolgens één keer in het model. Depth wordt met
`depth_scale` naar meters omgerekend, begrensd op `max_depth_m` en gedeeld door
die grens. Kanaal 2 is het geldigheidsmasker (`raw > 0`). Er is geen puntenwolk of
kunstmatig ingekleurd depth-beeld nodig. Depth gebruikt nearest-neighbour-resizing,
zodat ongeldige pixels en dieptesprongen niet worden uitgemiddeld.

`aligned_to_rgb` moet overeenkomen met het HDF5-attribuut. De live-adapter gebruikt
dezelfde instelling. Er worden geen onafhankelijke willekeurige RGB/depth-crops
uitgevoerd. ACT vereist dezelfde doelresolutie voor alle streams, omdat het de
featuremaps samenvoegt. Diffusion/flow hebben een aparte backbone per stream. ACT deelt één backbone
per modaliteit (dus één backbone voor alle RGB-camera's).

De inputarchitectuur en preprocessing verschillen van de oude modellen. Oude
checkpoints worden niet automatisch geconverteerd; train hiervoor nieuwe modellen.

## Acties worden tijdens laden afgeleid

Een bestand hoeft geen `action` te bevatten. De loader schrijft nooit naar HDF5.

- `action_source: auto`: gebruik bestaande acties; leid ze anders af uit joints.
- `action_source: next_qpos`: leid altijd af, ook als er al acties staan.
- `action_source: stored`: eis bestaande acties.
- `action_offset: 1`: doel is de volgende jointpositie, in dezelfde eenheden als qpos.

Met offset 1 geldt `action[t] = qpos[min(t+1, T-1)]`. De laatste positie wordt dus
herhaald. Een andere offset verandert de actiebetekenis en moet bewust worden gekozen.
Bij `auto` met aanwezige acties en bij `stored` worden de opgeslagen acties gebruikt;
de offset wordt daarop niet nogmaals toegepast.

De **chunkgrootte bepaalt hoeveel acties je achter elkaar aanbiedt**, niet wat
één actie betekent. De verschuiving gebeurt vóór het uitsnijden van chunks.
Ook de laatste actie van een chunk verwijst daardoor naar de juiste volgende
positie, als die nog in de episode bestaat. ACT markeert padding met `is_pad`;
diffusion/flow herhalen randwaarden volgens hun bestaande window-padding.
De normalisatiestatistieken gebruiken precies dezelfde actiebron, alleen op het
trainingsdeel van de dataset.

## Eén demo is een technische test

Gebruik voor echte held-out validatie minimaal twee demo's, liefst een grotere
verzameling. Met alleen de voorbeeldopname geef je expliciet toestemming om
hetzelfde bestand voor training én validatie te gebruiken:

- ACT: `--allow_single_episode` of `allow_single_episode: true` in `inputs.yaml`.
- Diffusion/flow: `task.allow_single_episode=true`.

Er verschijnt dan een melding: de validatie is geen onafhankelijke kwaliteitsmeting.

## Live-inference

ACT: `eval.py`; diffusion/flow: `eval_scara.py`, vanuit de bijbehorende projectmap.
De inputconfiguratie komt uit het checkpoint. Fysieke camera's krijgen standaard
serienummers `camera_top=130322273198` en `camera_bottom=123622270497`.
Je kunt die per camera overschrijven:

```text
--camera camera_top 130322273198 --camera camera_bottom 123622270497
```

De camera's worden parallel gelezen. Depth en RGB doorlopen dezelfde preprocessing
als tijdens training. De scripts starten standaard zonder robotbeweging; bestaande
bewegingsopties en controles zijn behouden. `--execute` schakelt beweging in.
Camera's en een robotverbinding zijn wel nodig voor live-observaties. De live
observatieroute leest joints na de cameraframes, zoals de oorspronkelijke
live-evaluatiescripts; dit is niet de interpolerende recorderpoller.

Hardwarebesturing is niet getest tijdens deze wijziging.

## Tests

Vanuit de repositoryroot `greenhouse-scara`, in de bestaande `act`-omgeving:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s src/greenhouse_scara_common/tests -v
```

De tien tests controleren actie-offsets en chunkranden, alleen RGB en gemengde
RGB/depth-selecties, preprocessing in loader en live-invoer, ongeldige/missende
data en dat HDF5-bestanden niet veranderen. Alle drie modellen worden ook met
twee RGB-D-camera's getest op loss, gradients naar elke stream en voorspelling.
Daarnaast zijn korte CPU-trainingsruns uitgevoerd voor alle drie op de aangeleverde
voorbeeldopname zonder `action`, met kleinere modellen en beeldresoluties.
De opgeslagen checkpoints zijn opnieuw geladen. Volledige training en fysieke
camera's/robot zijn hiermee nog niet gevalideerd.

## Opruiming van de kopieën

De greenhouse-projecten bevatten geen meegekopieerde simulatietaken of assets meer.
SCARA-training en offline/live-evaluatie houden dezelfde modulepaden voor checkpoints.
De mappen met model- en policycode blijven intact; gedeelde inputverwerking en tests
in deze map zijn eveneens behouden. De originele `scara_*`-projecten zijn ongewijzigd.
